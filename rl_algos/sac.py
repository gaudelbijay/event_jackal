import copy
import pickle
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rl_algos.base_rl_algo import BaseRLAlgo, ReplayBuffer

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianActor(nn.Module):
    def __init__(self, encoder, head, action_dim, device="cpu", init_weights=True):
        super(GaussianActor, self).__init__()
        self.encoder = encoder
        self.head = head
        self.device = device

        # Dynamically determine output_dim from head
        self.output_dim = self.get_head_output_dim()
        
        self.fc_mean = nn.Linear(self.output_dim, action_dim)
        self.fc_log_std = nn.Linear(self.output_dim, action_dim)

        if init_weights:
            self.head.apply(weights_init_)
            self.fc_mean.apply(weights_init_)
            self.fc_log_std.apply(weights_init_)

    def get_head_output_dim(self):
        for layer in reversed(list(self.head.modules())):
            if isinstance(layer, nn.Linear):
                return layer.out_features
        raise AttributeError("The head module does not contain any Linear layers.")

    def forward(self, state):
        a = self.encoder(state) if self.encoder else state
        a = self.head(a)
        mean = self.fc_mean(a)
        log_std = torch.clamp(self.fc_log_std(a), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)   # squash to [-1, 1]
        
        # Compute log_prob with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean)  # squash mean to [-1, 1]
        return y_t, log_prob, mean

class Critic(nn.Module):
    def __init__(self, encoder, head, init_weights=True):
        super(Critic, self).__init__()
        self.encoder1 = encoder
        self.head1 = head
        self.fc1 = nn.Linear(self.encoder1.feature_dim, 1)
        self.encoder2 = copy.deepcopy(encoder)
        self.head2 = copy.deepcopy(head)
        self.fc2 = nn.Linear(self.encoder2.feature_dim, 1)

        if init_weights:
            self.head1.apply(weights_init_)
            self.head2.apply(weights_init_)
            self.fc1.apply(weights_init_)
            self.fc2.apply(weights_init_)
            self.encoder1.apply(weights_init_)
            self.encoder2.apply(weights_init_)

    def forward(self, state, action):
        state1 = self.encoder1(state) if self.encoder1 else state
        sa1 = torch.cat([state1, action], 1)
        state2 = self.encoder2(state) if self.encoder2 else state
        sa2 = torch.cat([state2, action], 1)
        q1 = self.head1(sa1)
        q1 = self.fc1(q1)
        q2 = self.head2(sa2)
        q2 = self.fc2(q2)
        return q1, q2

    def Q1(self, state, action):
        state = self.encoder1(state) if self.encoder1 else state
        sa = torch.cat([state, action], 1)
        q1 = self.head1(sa)
        q1 = self.fc1(q1)
        return q1

class SAC(BaseRLAlgo):
    def __init__(
            self,
            actor,
            actor_optim,
            critic,
            critic_optim,
            action_range,
            device="cpu",
            gamma=0.99,
            n_step=4,
            tau=0.005,
            alpha=0.1,
            automatic_entropy_tuning=True,
            alpha_lr=1e-5,
            actor_max_norm=10.0,
            critic_max_norm=10.0,
            alpha_max_norm=2.0,
            q_value_clip=None,
    ):
        super(SAC, self).__init__(actor, actor_optim, critic, critic_optim, action_range, n_step, gamma, device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.actor_max_norm = actor_max_norm
        self.critic_max_norm = critic_max_norm
        self.alpha_max_norm = alpha_max_norm
        self.q_value_clip = q_value_clip

        # Handle actor wrapped in DataParallel if necessary
        if hasattr(actor, "module"):
            action_dim = actor.module.fc_mean.out_features
        else:
            action_dim = actor.fc_mean.out_features

        # Set target entropy based on the actor's action dimension
        if self.automatic_entropy_tuning:
            self.target_entropy = -float(action_dim)
            log_alpha_value = torch.log(torch.tensor(alpha, dtype=torch.float32))
            self.log_alpha = torch.tensor(
                [log_alpha_value], requires_grad=True, device=self.device, dtype=torch.float32
            )
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

    def train_rl(self, state, action, next_state, reward, not_done, gammas, collision_reward):
        # Convert action from replay buffer to the neural net expected range
        action_norm = self.unscale_action(action)

        with torch.no_grad():
            # Actor outputs actions in [-1, 1] range
            next_action, next_log_prob, _ = self.actor.sample(next_state) if not hasattr(self.actor, "module") else self.actor.module.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + not_done * (gammas * target_Q)
            if self.q_value_clip is not None:
                target_Q = torch.clamp(target_Q, min=-self.q_value_clip, max=self.q_value_clip)

        # Update critic
        current_Q1, current_Q2 = self.critic(state, action_norm)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_norm)
        critic_grad_norm = self.grad_norm(self.critic)
        self.critic_optimizer.step()

        # Update actor
        if hasattr(self.actor, "module"):
            action_pi, log_prob, _ = self.actor.module.sample(state)
        else:
            action_pi, log_prob, _ = self.actor.sample(state)
        Q1, Q2 = self.critic(state, action_pi)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_prob - Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_norm)
        actor_grad_norm = self.grad_norm(self.actor)
        self.actor_optimizer.step()

        # Update temperature parameter
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=self.alpha_max_norm)
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        self.log_alpha.data.clamp_(min=np.log(0.05))  

        # Update critic target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "Actor_grad_norm": actor_grad_norm,
            "Critic_grad_norm": critic_grad_norm,
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item() if self.automatic_entropy_tuning else self.alpha
        }

    def select_action(self, state, to_cpu=True):
        # Prepare state as tensor
        if isinstance(state, (list, tuple)):
            new_state = []
            for s in state:
                s_tensor = torch.FloatTensor(s).to(self.device)
                # If s_tensor is 2D (e.g. [stack_frame, feature_dim]) but we need a batch dimension,
                # then unsqueeze to get [1, stack_frame, feature_dim].
                if s_tensor.ndim == 2:
                    s_tensor = s_tensor.unsqueeze(0)
                new_state.append(s_tensor)
            state = tuple(new_state)
        else:
            state = torch.FloatTensor(state).to(self.device)
            if state.ndim < 2:
                state = state.unsqueeze(0)
            # Also check for 2D state (e.g. [stack_frame, feature_dim]) and add a batch dimension
            elif state.ndim == 2:
                state = state.unsqueeze(0)
        # Use underlying module if actor is wrapped in DataParallel
        if hasattr(self.actor, "module"):
            action, *_ = self.actor.module.sample(state)
        else:
            action, *_ = self.actor.sample(state)
        scaled_action = self.scale_action(action)
        if to_cpu:
            scaled_action = scaled_action.cpu().data.numpy().flatten()
        return scaled_action

    def save(self, dir, filename):
        super(SAC, self).save(dir, filename)
        # Ensure self.alpha is on CPU for serialization.
        alpha_to_save = self.alpha.cpu() if torch.is_tensor(self.alpha) else self.alpha
        with open(join(dir, filename + "_noise"), "wb") as f:
            pickle.dump(alpha_to_save, f)

    def load(self, dir, filename):
        super(SAC, self).load(dir, filename)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        with open(join(dir, filename + "_noise"), "rb") as f:
            loaded_alpha = pickle.load(f)
        # If loaded_alpha is a tensor, move it to the current device.
        self.alpha = loaded_alpha.to(self.device) if torch.is_tensor(loaded_alpha) else loaded_alpha

