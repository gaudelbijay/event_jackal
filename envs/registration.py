from gym.envs.registration import register

# DWA envs
register(
    id="dwa_param_continuous_events-v0",
    entry_point="envs.parameter_tuning_envs:DWAParamContinuousEvents"
)

# Motion control envs
register(
    id="motion_control_continuous_events-v0",
    entry_point="envs.motion_control_envs:MotionControlContinuousEvents"
)

# Place holder

register(
    id="motion_control_continuous_placeholder-v0",
    entry_point="envs.jackal_gazebo_placeholder:JackalGazeboEventsPlaceholder"
)

register(
    id="jackal_motion_control_continuous_events-v0",
    entry_point="envs.jackal_motion_control:RealMotionControlContinuousEvents"
)