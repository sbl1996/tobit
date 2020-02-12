import torch
from torch import Tensor
from torch import jit as jit, nn as nn
from torch.nn import Parameter
from torch.optim import Adam
from torch.utils.data import Dataset
from typing import Tuple


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


class LSTMFilterLayer(nn.Module):

    def __init__(self, obs_dim, hidden_dim, state_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(obs_dim, hidden_dim, 2, bidirectional=True)
        self.hidden2state = nn.Linear(hidden_dim * 2, state_dim)

    def forward(self, seqs):
        lstm_out, _ = self.lstm(seqs)
        states = self.hidden2state(lstm_out)
        return states


class GenDataset(Dataset):

    @staticmethod
    def collate_fn(batch):
        xs, ys = zip(*batch)
        xs = torch.stack(xs, dim=1)
        ys = torch.stack(ys, dim=1)
        return xs, ys

    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def __getitem__(self, item):
        return self.gen()

    def __len__(self):
        return 100000


class LSTMFilter:

    def __init__(self, obs_dim, state_dim, hidden_dim=6, bp_through_time=1):
        self.model = LSTMFilterLayer(obs_dim, hidden_dim, state_dim)
        self.obs_dim = obs_dim

        self.bp_through_time = bp_through_time

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        self._iter = 0

    def train(self, train_iterator, num_iter, val_gen=None, val_per_iter=50):

        it = train_iterator

        for i in range(self._iter + 1, self._iter + num_iter + 1):
            x, y = next(it)

            self.model.train()
            states = self.model(x)

            loss = self.criterion(states, y)
            loss.backward()

            if i % self.bp_through_time == 0:
                self.optimizer.step()
                self.model.zero_grad()

            if val_gen and i % val_per_iter == 0:
                x, y = val_gen()

                self.model.eval()
                states = self.predict(x)
                loss2 = self.criterion(states, y).item()
                print("%5d: %3.4f  %2.4f" % (i, loss.item(), loss2))

            self._iter += 1

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            yt = self.model(x[:, None, :])[:, 0, :]
        return yt