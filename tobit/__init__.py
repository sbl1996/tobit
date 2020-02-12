import os
import torch.utils.cpp_extension

torch.utils.cpp_extension.load(
    name="tkf",
    sources=[os.path.join(os.path.dirname(__file__), "cpp", "op.cpp")],
    extra_ldflags=[],
    is_python_module=False,
    verbose=False,
)

from tobit.standard import KalmanFilter
from tobit.tobit import TobitKalmanFilter


# class TobitKalmanFilterM(torch.nn.Module):
#     """
#     A: transition_matrices
#     H: observation_matrices
#     Q: transition_covariance
#     R: observation_covariance
#     x0: initial_state_mean
#     P0: initial_state_covariance
#     """
#
#     def __init__(self, lower_limits, upper_limits,
#                  transition_matrices=None, observation_matrices=None,
#                  transition_covariance=None, observation_covariance=None,
#                  initial_state_mean=None, initial_state_covariance=None,
#                  n_dim_obs=None, n_dim_state=None):
#         super().__init__()
#         self.lower_limits = atleast_1d(lower_limits)
#         self.upper_limits = atleast_1d(upper_limits)
#         if transition_matrices is None:
#             assert n_dim_state
#             transition_matrices = torch.eye(n_dim_state, dtype=torch.float32)
#         self.transition_matrices = atleast_2d(transition_matrices)
#
#         if observation_matrices is None:
#             assert n_dim_obs
#             assert n_dim_state
#             observation_matrices = torch.eye(n_dim_obs, n_dim_state, dtype=torch.float32)
#         self.observation_matrices = atleast_2d(observation_matrices).type_as(transition_matrices)
#         self._n_dim_obs, self._n_dim_state = self.observation_matrices.shape
#
#         if transition_covariance is None:
#             transition_covariance = torch.eye(n_dim_state, dtype=torch.float32)
#         self.transition_covariance = atleast_2d(transition_covariance).type_as(transition_matrices)
#
#         if observation_covariance is None:
#             observation_covariance = torch.eye(n_dim_obs, dtype=torch.float32)
#         self.observation_covariance = atleast_2d(observation_covariance).type_as(transition_matrices)
#
#         if initial_state_mean is None:
#             initial_state_mean = torch.zeros(n_dim_state)
#         self.initial_state_mean = atleast_1d(initial_state_mean).type_as(transition_matrices)
#
#         if initial_state_covariance is None:
#             initial_state_covariance = torch.eye(n_dim_state)
#         self.initial_state_covariance = atleast_2d(initial_state_covariance).type_as(transition_matrices)
#
#     def forward(self, measurements):
#         measurements = atleast_2d(measurements)
#
#         T = measurements.shape[0]
#         state_means = [self.initial_state_mean]
#         state_covariances = [self.initial_state_covariance]
#         for t in range(T):
#             x, P = predict(state_means[-1], state_covariances[-1], self.transition_matrices, self.transition_covariance)
#             x, P = tobit_update(
#                 measurements[t], x, P, self.observation_matrices, self.observation_covariance,
#                 self.lower_limits, self.upper_limits)
#             state_means.append(x)
#             state_covariances.append(P)
#
#         state_means = torch.stack(state_means[1:])
#         state_covariances = torch.stack(state_covariances[1:])
#         return state_means, state_covariances
