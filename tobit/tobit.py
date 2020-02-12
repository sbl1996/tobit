import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor

from tobit.standard import predict
from tobit.ops import cdf, pdf, atleast_1d, atleast_2d, batched_diag


@torch.jit.script
def tobit_update(y, x, P, H, R, Tl, Tu):
    r = torch.sqrt(torch.diagonal(R)) + 1e-4
    z = H @ x
    zl = (Tl - z) / r
    zu = (Tu - z) / r
    cpl = cdf(zl)
    cpu = cdf(zu)
    ppl = pdf(zl)
    ppu = pdf(zu)
    p = cpu - cpl + 1e-4
    Pun = torch.diag(p)
    l = (ppu - ppl) / p
    c = zl * ppl - zu * ppu
    Ey = p * (z - r * l) + cpl * Tl + (1 - cpu) * Tu
    R = R @ torch.diag(c / p + 1 - l ** 2)
    R1 = torch.chain_matmul(P, H.t(), Pun)
    R2 = torch.chain_matmul(Pun, H, R1) + R
    K = R1 @ torch.inverse(R2)
    x = x + K @ (y - Ey)
    P = P - torch.chain_matmul(K, Pun, H, P)
    return x, P


class TobitKalmanFilter:
    """
    A: transition_matrices
    H: observation_matrices
    Q: transition_covariance
    R: observation_covariance
    x0: initial_state_mean
    P0: initial_state_covariance
    """

    def __init__(self, lower_limits, upper_limits,
                 transition_matrices=None, observation_matrices=None,
                 transition_covariance=None, observation_covariance=None,
                 initial_state_mean=None, initial_state_covariance=None,
                 n_dim_obs=None, n_dim_state=None, cpp=True):
        self.lower_limits = atleast_1d(lower_limits)
        self.upper_limits = atleast_1d(upper_limits)
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

        if initial_state_mean is None:
            initial_state_mean = torch.zeros(n_dim_state)
        self.initial_state_mean = atleast_1d(initial_state_mean).type_as(transition_matrices)

        if initial_state_covariance is None:
            initial_state_covariance = torch.eye(n_dim_state)
        self.initial_state_covariance = atleast_2d(initial_state_covariance).type_as(transition_matrices)

        self.cpp = cpp

    def filter(self, measurements):
        measurements = atleast_2d(measurements)

        if self.cpp:
            return torch.ops.tkf.tobit_kalman_filter(
                self.lower_limits, self.upper_limits,
                self.initial_state_mean, self.initial_state_covariance,
                self.transition_matrices, self.observation_matrices,
                self.transition_covariance, self.observation_covariance, measurements
            )

        T = measurements.shape[0]
        state_means = [self.initial_state_mean]
        state_covariances = [self.initial_state_covariance]
        for t in range(T):
            x, P = predict(state_means[-1], state_covariances[-1], self.transition_matrices, self.transition_covariance)
            x, P = tobit_update(
                measurements[t], x, P, self.observation_matrices, self.observation_covariance,
                self.lower_limits, self.upper_limits)
            state_means.append(x)
            state_covariances.append(P)

        state_means = torch.stack(state_means[1:])
        state_covariances = torch.stack(state_covariances[1:])
        return state_means, state_covariances


class LSTMCell(jit.ScriptModule):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size), requires_grad=True)
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size), requires_grad=True)
        self.bias_ih = Parameter(torch.randn(4 * hidden_size), requires_grad=True)
        self.bias_hh = Parameter(torch.randn(4 * hidden_size), requires_grad=True)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LayerNormLSTMCell(jit.ScriptModule):

    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases

        ln = nn.LayerNorm

        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class TobitKalmanLSTMLayer(jit.ScriptModule):

    def __init__(self, obs_dim, hidden_dim, state_dim, H, Tl, Tu, layer_norm=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim

        cell = LayerNormLSTMCell if layer_norm else LSTMCell

        self.cell_mean = cell(state_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, state_dim)

        self.cell_F = cell(state_dim, hidden_dim)
        self.fc_F = nn.Linear(hidden_dim, state_dim * state_dim)

        self.cell_Q = cell(state_dim, hidden_dim)
        self.fc_Q = nn.Linear(hidden_dim, state_dim)

        self.cell_R = cell(state_dim, hidden_dim)
        self.fc_R = nn.Linear(hidden_dim, obs_dim)

        self.H = nn.Parameter(H, requires_grad=False)
        self.Tl = nn.Parameter(Tl, requires_grad=False)
        self.Tu = nn.Parameter(Tu, requires_grad=False)

    @jit.script_method
    def forward(self, x, y, P, state_y, state_F, state_Q, state_R):
        # type: (Tensor, Tensor, Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        batch_size = x.size(1)
        xs = x.unbind(0)
        outputs1 = torch.jit.annotate(List[Tensor], [])
        outputs2 = torch.jit.annotate(List[Tensor], [])
        for i in range(len(xs)):
            y, state_y = self.cell_mean(y, state_y)
            y = self.fc_mean(y)
            outputs1 += [y]

            F, state_F = self.cell_F(y, state_F)
            F = self.fc_F(F).view(batch_size, self.state_dim, self.state_dim)

            Q, state_Q = self.cell_Q(y, state_Q)
            Q = self.fc_Q(Q).exp()

            R, state_R = self.cell_R(y, state_R)
            R = self.fc_R(R).exp()

            ys = []
            Ps = []
            for b in range(batch_size):
                Fb = F[b]
                Qb = torch.diag(Q[b])
                Rb = torch.diag(R[b])
                Pb = torch.chain_matmul(Fb, P[b], Fb.t()) + Qb
                yb, Pb = tobit_update(xs[i][b], y[b], Pb, self.H, Rb, self.Tl, self.Tu)
                ys.append(yb)
                Ps.append(Pb)
            y = torch.stack(ys)
            P = torch.stack(Ps)
            outputs2 += [y]
        return torch.stack(outputs1), torch.stack(outputs2)


class TobitKalmanLSTM(nn.Module):

    def __init__(self, obs_dim, hidden_dim, state_dim, H, Tl, Tu, layer_norm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        layer = TobitKalmanLSTMLayer(obs_dim, hidden_dim, state_dim, H, Tl, Tu, layer_norm)
        self.layer = torch.jit.script(layer)

    def forward(self, x, y, P):
        batch_size = x.size(1)
        y = y[None].expand(batch_size, -1)
        P = P[None].expand(batch_size, -1, -1)
        h0 = x.new_zeros(batch_size, self.hidden_dim)
        c0 = x.new_zeros(batch_size, self.hidden_dim)
        state_y = (h0, c0)
        state_F = (h0, c0)
        state_Q = (h0, c0)
        state_R = (h0, c0)
        return self.layer(x, y, P, state_y, state_F, state_Q, state_R)
