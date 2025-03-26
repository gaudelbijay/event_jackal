# model_based.py

import copy
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from rl_algos.base_rl_algo import BaseRLAlgo


class Model(nn.Module):
    """
    Predictive Model for Event Camera-based Reinforcement Learning.

    This model predicts the next state, reward, and done signal given the current state and action.
    It can operate deterministically or stochastically.
    """
    def __init__(self, encoder, head, state_dim, deterministic=False):
        """
        Initialize the Model.

        Args:
            encoder (nn.Module): Encoder network to process the state.
            head (nn.Module): Head network to process the concatenated state and action.
            state_dim (tuple): Tuple containing (history_length, state_dimension).
            deterministic (bool): If True, the model predicts deterministically.
        """
        super(Model, self).__init__()
        self.encoder = encoder
        self.head = head
        self.deterministic = deterministic
        self.history_length, self.state_dim = state_dim

        if not deterministic:
            self.state_dim *= 2  # mean and logvar

        self.state_fc = nn.Sequential(
            nn.Linear(head.feature_dim, self.state_dim),
            nn.Tanh()
        )

        self.reward_fc = nn.Linear(head.feature_dim, 1)

        self.done_fc = nn.Linear(head.feature_dim, 1)

    def forward(self, state, action):
        """
        Forward pass to predict the next state, reward, and done signal.

        Args:
            state (tuple): Current state, typically a tuple of (event_state, goal_action_state).
            action (torch.Tensor): Current action.

        Returns:
            tuple: Predicted next state, reward, and done signal.
        """
        s = self.encoder(state) if self.encoder else state
        sa = torch.cat([s, action], dim=1)
        x = self.head(sa)
        s_pred = self.state_fc(x)
        r_pred = self.reward_fc(x)
        d_pred = torch.sigmoid(self.done_fc(x))
        if not self.deterministic:
            mean = s_pred[..., self.state_dim // 2:]
            logvar = s_pred[..., :self.state_dim // 2]
            s_pred = torch.cat([mean, logvar], dim=1)
        return s_pred, r_pred, d_pred

    def sample(self, state, action):
        """
        Sample the next state based on the model's predictions.

        Args:
            state (torch.Tensor): Current state.
            action (torch.Tensor): Current action.

        Returns:
            tuple: Sampled next state, reward, and done signal.
        """
        s_pred, r_pred, d_pred = self.forward(state, action)

        if self.deterministic:
            if self.history_length > 1:
                next_state = torch.cat([state[:, 1:, :], s_pred[:, None, :]], dim=1)
                return next_state, r_pred, d_pred
            else:
                return s_pred[:, None, :], r_pred, d_pred
        else:
            mean = s_pred[..., self.state_dim // 2:]
            logvar = s_pred[..., :self.state_dim // 2]
            std = torch.exp(logvar)
            recon_dist = Normal(mean, std)
            sampled_state = recon_dist.rsample()

            if self.history_length > 1:
                next_state = torch.cat([state[:, 1:, :], sampled_state[:, None, :]], dim=1)
                return next_state, r_pred, d_pred
            else:
                return sampled_state[:, None, :], r_pred, d_pred


class SMCPRLAlgo(BaseRLAlgo):
    """
    Stochastic Model-based Control with Particle-based Reinforcement Learning (SMCPRL) Algorithm.

    Combines model predictive control with reinforcement learning using a learned model to simulate future states.
    """
    def __init__(self, model, model_optimizer, *args, horizon=5, num_particle=1024, model_update_per_step=5, 
                 n_simulated_update=5, gradient_clip=0.5, noise_decay=0.99, **kw_args):
        """
        Initialize the SMCPRLAlgo.

        Args:
            model (Model): Learned environment model.
            model_optimizer (torch.optim.Optimizer): Optimizer for the model.
            horizon (int): Planning horizon for MPC.
            num_particle (int): Number of particles for simulation.
            model_update_per_step (int): Number of model updates per training step.
            n_simulated_update (int): Number of simulated updates per training step.
            gradient_clip (float): Gradient clipping threshold.
            noise_decay (float): Decay rate for exploration noise.
        """
        self.model = model
        self.model_optimizer = model_optimizer
        self.horizon = horizon
        self.num_particle = num_particle
        self.model_update_per_step = model_update_per_step
        self.n_simulated_update = n_simulated_update
        self.loss_function = nn.MSELoss()
        self.gradient_clip = gradient_clip
        self.noise_decay = noise_decay
        self.current_noise = 1.0
        super().__init__(*args, **kw_args)

    def compute_model_loss(self, state, action, next_state, reward, done):
        """
        Compute the loss for the learned model.

        Args:
            state (torch.Tensor): Current state.
            action (torch.Tensor): Current action.
            next_state (torch.Tensor): True next state.
            reward (torch.Tensor): True reward.
            done (torch.Tensor): Done signal.

        Returns:
            torch.Tensor: Total model loss.
        """
        pred_next_state, pred_reward, pred_done = self.model(state, action)

        if self.model.deterministic:
            state_loss = self.loss_function(pred_next_state, next_state[:, -1, :])
        else:
            mean = pred_next_state[..., self.model.state_dim // 2:]
            logvar = pred_next_state[..., :self.model.state_dim // 2]
            recon_dist = Normal(mean, torch.exp(logvar))
            state_loss = -recon_dist.log_prob(next_state[:, -1, :]).sum(dim=-1).mean()

        reward_loss = self.loss_function(pred_reward, reward)
        done_loss = F.binary_cross_entropy(pred_done, done)

        total_loss = state_loss + reward_loss + done_loss
        return total_loss

    def train_model(self, replay_buffer, batch_size=256):
        """
        Train the environment model using samples from the replay buffer.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer containing experience tuples.
            batch_size (int): Number of samples per batch.

        Returns:
            dict: Dictionary containing model loss and gradient norm.
        """
        state, action, next_state, reward, not_done, *_ = replay_buffer.sample(batch_size)
        done = 1 - not_done

        action = (action - self._action_bias) / self._action_scale

        loss = self.compute_model_loss(state, action, next_state, reward, done)

        self.model_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.model_optimizer.step()

        return {
            "Model_loss": loss.item(),
            "Model_grad_norm": self.grad_norm(self.model)
        }

    def simulate_transition(self, replay_buffer, batch_size=256):
        """
        Simulate a transition using the learned model.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer for sampling initial states.
            batch_size (int): Number of samples per batch.

        Returns:
            tuple: Simulated state, action, next_state, reward, not_done, gammas.
        """
        state, action, next_state, reward, not_done, *_ = replay_buffer.sample(batch_size)
        total_reward = torch.zeros(reward.shape).to(self.device)
        not_done = torch.ones(reward.shape).to(self.device)
        gammas = 1

        with torch.no_grad():
            for i in range(self.horizon):
                next_action = self.actor_target(state)
                next_action += torch.randn_like(next_action, dtype=torch.float32) * self.exploration_noise
                if i == 0:
                    action = next_action

                next_state, r, d = self.model.sample(state, next_action)

                reward = r
                not_done *= (1 - d)
                gammas *= self.gamma ** not_done
                reward = (reward - replay_buffer.mean) / replay_buffer.std
                total_reward = reward + total_reward * gammas

                state = next_state

        return state, action, next_state, reward, not_done, gammas

    def train(self, replay_buffer, batch_size=256):
        """
        Train both the reinforcement learning policy and the environment model.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer containing experience tuples.
            batch_size (int): Number of samples per batch.

        Returns:
            dict: Dictionary containing aggregated loss information.
        """
        rl_loss_info = super().train(replay_buffer, batch_size)

        for _ in range(self.model_update_per_step):
            model_loss_info = self.train_model(replay_buffer, batch_size)

        simulated_rl_loss_infos = []
        for _ in range(self.n_simulated_update):
            state, action, next_state, reward, not_done, gammas = self.simulate_transition(replay_buffer, batch_size)
            simulated_rl_loss_info = self.train_rl(state, action, next_state, reward, not_done, gammas, None)
            simulated_rl_loss_infos.append(simulated_rl_loss_info)

        simulated_rl_loss_info = {}
        for k in simulated_rl_loss_infos[0].keys():
            simulated_rl_loss_info["simulated_" + k] = np.mean([li[k] for li in simulated_rl_loss_infos if li[k] is not None])

        loss_info = {**rl_loss_info, **model_loss_info, **simulated_rl_loss_info}
        return loss_info

    def select_action(self, state):
        """
        Select an action using the policy with particle-based simulations.

        Args:
            state (tuple): Current state, typically a tuple of (event_state, goal_action_state).

        Returns:
            np.ndarray: Selected action.
        """
        if self.exploration_noise >= 0:
            assert len(state.shape) == 2, "Does not support batched action selection!"

            state = torch.FloatTensor(state).to(self.device)[None, ...]
            s = state.repeat(self.num_particle, 1, 1).clone()

            r = torch.zeros((self.num_particle, 1)).to(self.device)
            gamma = torch.ones((self.num_particle, 1)).to(self.device)

            exploration_noise = self.current_noise * self.exploration_noise

            with torch.no_grad():
                for i in range(self.horizon):
                    a = self.actor(s)
                    a += torch.randn_like(a, dtype=torch.float32) * exploration_noise
                    if i == 0:
                        a0 = a
                    s, r_step, d_step = self.model.sample(s, a)
                    r += r_step * gamma
                    gamma *= (1 - d_step)

            q_value = self.critic.Q1(s, a)
            r += q_value * gamma

            logit_r = F.softmax(r, dim=-1).view(-1)
            n = Categorical(logit_r).sample()
            selected_action = a0[n]

            self.current_noise *= self.noise_decay

            return selected_action.cpu().numpy()
        else:
            return super().select_action(state)

    def save(self, dir, filename):
        """
        Save the policy and model parameters.

        Args:
            dir (str): Directory to save the parameters.
            filename (str): Base filename for the saved parameters.
        """
        super().save(dir, filename)
        self.model.to("cpu")
        with open(join(dir, filename + "_model.pkl"), "wb") as f:
            pickle.dump(self.model.state_dict(), f)
        self.model.to(self.device)

    def load(self, dir, filename):
        """
        Load the policy and model parameters.

        Args:
            dir (str): Directory from which to load the parameters.
            filename (str): Base filename of the saved parameters.
        """
        super().load(dir, filename)
        with open(join(dir, filename + "_model.pkl"), "rb") as f:
            self.model.load_state_dict(pickle.load(f))


class DynaRLAlgo(BaseRLAlgo):
    """
    Dyna-Q Reinforcement Learning Algorithm with Model-based Simulations.

    Combines model-free RL with model-based planning to improve sample efficiency.
    """
    def __init__(self, model, model_optimizer, *args, model_update_per_step=5, n_simulated_update=5, **kw_args):
        """
        Initialize the DynaRLAlgo.

        Args:
            model (Model): Learned environment model.
            model_optimizer (torch.optim.Optimizer): Optimizer for the model.
            model_update_per_step (int): Number of model updates per training step.
            n_simulated_update (int): Number of simulated updates per training step.
        """
        self.model = model
        self.model_optimizer = model_optimizer
        self.model_update_per_step = model_update_per_step
        self.n_simulated_update = n_simulated_update
        self.loss_function = nn.MSELoss()
        super().__init__(*args, **kw_args)

    def train_model(self, replay_buffer, batch_size=256):
        """
        Train the environment model using samples from the replay buffer.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer containing experience tuples.
            batch_size (int): Number of samples per batch.

        Returns:
            dict: Dictionary containing model loss and gradient norm.
        """
        state, action, next_state, reward, not_done, *_ = replay_buffer.sample(batch_size)
        done = 1 - not_done

        action = (action - self._action_bias) / self._action_scale

        pred_next_state, pred_reward, pred_done = self.model(state, action)

        if self.model.deterministic:
            state_loss = self.loss_function(pred_next_state, next_state[:, -1, :])
        else:
            mean = pred_next_state[..., self.model.state_dim // 2:]
            logvar = pred_next_state[..., :self.model.state_dim // 2]
            recon_dist = Normal(mean, torch.exp(logvar))
            state_loss = -recon_dist.log_prob(next_state[:, -1, :]).sum(dim=-1).mean()

        reward_loss = self.loss_function(pred_reward, reward)
        done_loss = F.binary_cross_entropy(pred_done, done)

        loss = state_loss + reward_loss + done_loss

        self.model_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        self.model_optimizer.step()

        return {
            "Model_loss": loss.item(),
            "Model_grad_norm": self.grad_norm(self.model)
        }

    def simulate_transition(self, replay_buffer, batch_size=256):
        """
        Simulate a transition using the learned model.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer for sampling initial states.
            batch_size (int): Number of samples per batch.

        Returns:
            tuple: Simulated state, action, next_state, reward, not_done, gammas.
        """
        state, action, next_state, reward, not_done, *_ = replay_buffer.sample(batch_size)
        total_reward = torch.zeros(reward.shape).to(self.device)
        not_done = torch.ones(reward.shape).to(self.device)
        gammas = 1

        with torch.no_grad():
            for i in range(self.horizon):
                next_action = self.actor_target(state)
                next_action += torch.randn_like(next_action, dtype=torch.float32) * self.exploration_noise
                if i == 0:
                    action = next_action

                next_state, r, d = self.model.sample(state, next_action)

                reward = r
                not_done *= (1 - d)
                gammas *= self.gamma ** not_done
                reward = (reward - replay_buffer.mean) / replay_buffer.std
                total_reward = reward + total_reward * gammas

                state = next_state

        return state, action, next_state, reward, not_done, gammas

    def train(self, replay_buffer, batch_size=256):
        """
        Train both the reinforcement learning policy and the environment model.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer containing experience tuples.
            batch_size (int): Number of samples per batch.

        Returns:
            dict: Dictionary containing aggregated loss information.
        """
        rl_loss_info = super().train(replay_buffer, batch_size)

        for _ in range(self.model_update_per_step):
            model_loss_info = self.train_model(replay_buffer, batch_size)

        simulated_rl_loss_infos = []
        for _ in range(self.n_simulated_update):
            state, action, next_state, reward, not_done, gammas = self.simulate_transition(replay_buffer, batch_size)
            simulated_rl_loss_info = self.train_rl(state, action, next_state, reward, not_done, gammas, None)
            simulated_rl_loss_infos.append(simulated_rl_loss_info)

        simulated_rl_loss_info = {}
        for k in simulated_rl_loss_infos[0].keys():
            simulated_rl_loss_info["simulated_" + k] = np.mean([li[k] for li in simulated_rl_loss_infos if li[k] is not None])

        loss_info = {**model_loss_info, **simulated_rl_loss_info}
        return loss_info

    def save(self, dir, filename):
        """
        Save the policy and model parameters.

        Args:
            dir (str): Directory to save the parameters.
            filename (str): Base filename for the saved parameters.
        """
        super().save(dir, filename)
        self.model.to("cpu")
        with open(join(dir, filename + "_model.pkl"), "wb") as f:
            pickle.dump(self.model.state_dict(), f)
        self.model.to(self.device)

    def load(self, dir, filename):
        """
        Load the policy and model parameters.

        Args:
            dir (str): Directory from which to load the parameters.
            filename (str): Base filename of the saved parameters.
        """
        super().load(dir, filename)
        with open(join(dir, filename + "_model.pkl"), "rb") as f:
            self.model.load_state_dict(pickle.load(f))


class MBPORLAlgo(BaseRLAlgo):
    """
    Model-Based Policy Optimization (MBPO) Reinforcement Learning Algorithm.

    Utilizes a learned model to generate synthetic transitions for policy optimization.
    """
    def __init__(self, model, model_optimizer, *args, horizon=5, num_particle=1024, 
                 model_update_per_step=5, n_simulated_update=5, gradient_clip=0.5, noise_decay=0.99, **kw_args):
        """
        Initialize the MBPORLAlgo.

        Args:
            model (Model): Learned environment model.
            model_optimizer (torch.optim.Optimizer): Optimizer for the model.
            horizon (int): Planning horizon for MPC.
            num_particle (int): Number of particles for simulation.
            model_update_per_step (int): Number of model updates per training step.
            n_simulated_update (int): Number of simulated updates per training step.
            gradient_clip (float): Gradient clipping threshold.
            noise_decay (float): Decay rate for exploration noise.
        """
        self.model = model
        self.model_optimizer = model_optimizer
        self.horizon = horizon
        self.num_particle = num_particle
        self.model_update_per_step = model_update_per_step
        self.n_simulated_update = n_simulated_update
        self.loss_function = nn.MSELoss()
        self.gradient_clip = gradient_clip
        self.noise_decay = noise_decay
        self.current_noise = 1.0
        super().__init__(*args, **kw_args)

    def compute_model_loss(self, state, action, next_state, reward, done):
        """
        Compute the loss for the learned model.

        Args:
            state (torch.Tensor): Current state.
            action (torch.Tensor): Current action.
            next_state (torch.Tensor): True next state.
            reward (torch.Tensor): True reward.
            done (torch.Tensor): Done signal.

        Returns:
            torch.Tensor: Total model loss.
        """
        pred_next_state, pred_reward, pred_done = self.model(state, action)

        if self.model.deterministic:
            state_loss = self.loss_function(pred_next_state, next_state[:, -1, :])
        else:
            mean = pred_next_state[..., self.model.state_dim // 2:]
            logvar = pred_next_state[..., :self.model.state_dim // 2]
            recon_dist = Normal(mean, torch.exp(logvar))
            state_loss = -recon_dist.log_prob(next_state[:, -1, :]).sum(dim=-1).mean()

        reward_loss = self.loss_function(pred_reward, reward)
        done_loss = F.binary_cross_entropy(pred_done, done)

        total_loss = state_loss + reward_loss + done_loss
        return total_loss

    def train_model(self, replay_buffer, batch_size=256):
        """
        Train the environment model using samples from the replay buffer.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer containing experience tuples.
            batch_size (int): Number of samples per batch.

        Returns:
            dict: Dictionary containing model loss and gradient norm.
        """
        state, action, next_state, reward, not_done, *_ = replay_buffer.sample(batch_size)
        done = 1 - not_done

        action = (action - self._action_bias) / self._action_scale

        loss = self.compute_model_loss(state, action, next_state, reward, done)

        self.model_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.model_optimizer.step()

        return {
            "Model_loss": loss.item(),
            "Model_grad_norm": self.grad_norm(self.model)
        }

    def simulate_transition(self, replay_buffer, batch_size=256):
        """
        Simulate a transition using the learned model.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer for sampling initial states.
            batch_size (int): Number of samples per batch.

        Returns:
            tuple: Simulated state, action, next_state, reward, not_done, gammas.
        """
        state, action, next_state, reward, not_done, *_ = replay_buffer.sample(batch_size)
        total_reward = torch.zeros(reward.shape).to(self.device)
        not_done = torch.ones(reward.shape).to(self.device)
        gammas = 1

        with torch.no_grad():
            for i in range(self.horizon):
                next_action = self.actor_target(state)
                next_action += torch.randn_like(next_action, dtype=torch.float32) * self.exploration_noise
                if i == 0:
                    action = next_action

                next_state, r, d = self.model.sample(state, next_action)

                reward = r
                not_done *= (1 - d)
                gammas *= self.gamma ** not_done
                reward = (reward - replay_buffer.mean) / replay_buffer.std
                total_reward = reward + total_reward * gammas

                state = next_state

        return state, action, next_state, reward, not_done, gammas

    def train(self, replay_buffer, batch_size=256):
        """
        Train both the reinforcement learning policy and the environment model.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer containing experience tuples.
            batch_size (int): Number of samples per batch.

        Returns:
            dict: Dictionary containing aggregated loss information.
        """
        rl_loss_info = super().train(replay_buffer, batch_size)

        for _ in range(self.model_update_per_step):
            model_loss_info = self.train_model(replay_buffer, batch_size)

        simulated_rl_loss_infos = []
        for _ in range(self.n_simulated_update):
            state, action, next_state, reward, not_done, gammas = self.simulate_transition(replay_buffer, batch_size)
            simulated_rl_loss_info = self.train_rl(state, action, next_state, reward, not_done, gammas, None)
            simulated_rl_loss_infos.append(simulated_rl_loss_info)

        simulated_rl_loss_info = {}
        for k in simulated_rl_loss_infos[0].keys():
            simulated_rl_loss_info["simulated_" + k] = np.mean([li[k] for li in simulated_rl_loss_infos if li[k] is not None])

        loss_info = {**model_loss_info, **simulated_rl_loss_info}
        return loss_info

    def save(self, dir, filename):
        """
        Save the policy and model parameters.

        Args:
            dir (str): Directory to save the parameters.
            filename (str): Base filename for the saved parameters.
        """
        super().save(dir, filename)
        self.model.to("cpu")
        with open(join(dir, filename + "_model.pkl"), "wb") as f:
            pickle.dump(self.model.state_dict(), f)
        self.model.to(self.device)

    def load(self, dir, filename):
        """
        Load the policy and model parameters.

        Args:
            dir (str): Directory from which to load the parameters.
            filename (str): Base filename of the saved parameters.
        """
        super().load(dir, filename)
        with open(join(dir, filename + "_model.pkl"), "rb") as f:
            self.model.load_state_dict(pickle.load(f))
