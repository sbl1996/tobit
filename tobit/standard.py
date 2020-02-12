import torch
from tobit.ops import atleast_2d, atleast_1d


def predict(x, P, A, Q):
    x = A @ x
    P = torch.chain_matmul(A, P, A.t()) + Q
    return x, P


def update(y, x, P, H, R):
    K = torch.chain_matmul(P, H.t(), torch.inverse(torch.chain_matmul(H, P, H.t()) + R))
    x = x + K @ (y - H @ x)
    P = P - torch.chain_matmul(K, H, P)
    return x, P


class KalmanFilter:
    """
    A: transition_matrices
    H: observation_matrices
    Q: transition_covariance
    R: observation_covariance
    x0: initial_state_mean
    P0: initial_state_covariance
    """

    def __init__(self, transition_matrices=None, observation_matrices=None,
                 transition_covariance=None, observation_covariance=None,
                 initial_state_mean=None, initial_state_covariance=None,
                 n_dim_obs=None, n_dim_state=None, cpp=False):
        if transition_matrices is None:
            assert n_dim_state
            transition_matrices = torch.eye(n_dim_state, dtype=torch.float32)
        self.transition_matrices = atleast_2d(transition_matrices)

        if observation_matrices is None:
            assert n_dim_obs
            assert n_dim_state
            observation_matrices = torch.eye(n_dim_obs, n_dim_state, dtype=torch.float32)
        self.observation_matrices = atleast_2d(observation_matrices).type_as(transition_matrices)
        self._n_dim_obs, self._n_dim_state = self.observation_matrices.shape

        if transition_covariance is None:
            transition_covariance = torch.eye(n_dim_state, dtype=torch.float32)
        self.transition_covariance = atleast_2d(transition_covariance).type_as(transition_matrices)

        if observation_covariance is None:
            observation_covariance = torch.eye(n_dim_obs, dtype=torch.float32)
        self.observation_covariance = atleast_2d(observation_covariance).type_as(transition_matrices)

        self.initial_state_mean = atleast_1d(initial_state_mean).type_as(transition_matrices)
        self.initial_state_covariance = atleast_2d(initial_state_covariance).type_as(transition_matrices)

        self.cpp = cpp

    def filter(self, measurements):
        measurements = atleast_2d(measurements)

        if self.cpp:
            return torch.ops.tkf.kalman_filter(
                self.initial_state_mean, self.initial_state_covariance,
                self.transition_matrices, self.observation_matrices,
                self.transition_covariance, self.observation_covariance, measurements
            )

        T = measurements.shape[0]
        state_means = [self.initial_state_mean]
        state_covariances = [self.initial_state_covariance]
        for t in range(T):
            x, P = predict(state_means[-1], state_covariances[-1], self.transition_matrices, self.transition_covariance)
            x, P = update(measurements[t], x, P, self.observation_matrices, self.observation_covariance)
            state_means.append(x)
            state_covariances.append(P)

        state_means = torch.stack(state_means[1:])
        state_covariances = torch.stack(state_covariances[1:])
        return state_means, state_covariances