import torch
import torch.nn as nn

# TODO: implement your model here
#       - You might want to create two classes, one for the LSTM cell and one for the LSTM model
#		- There is a naive and fast implementation possibility. Try to make your model efficient..


class LSTMCell(nn.Module):
    """Efficient LSTM cell - computes all 4 gates in one matrix multiply"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # All 4 gates combined into one Linear (the fast implementation)
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x, h, s):
        """
        x: (batch, input_size)
        h: (batch, hidden_size)
        s: (batch, hidden_size)
        """
        # One operation for all gates
        gates = self.W_x(x) + self.W_h(h)

        # Split into forget, input, candidate, output gates
        phi, iota, s_candidate, omega = gates.chunk(4, dim=-1)

        phi         = torch.sigmoid(phi)          # forget gate
        iota        = torch.sigmoid(iota)         # input gate
        s_candidate = torch.tanh(s_candidate)     # cell candidate
        omega       = torch.sigmoid(omega)        # output gate

        # Update cell and hidden state
        s_new = phi * s + iota * s_candidate
        h_new = omega * torch.tanh(s_new)

        return h_new, s_new


class LSTMLayer(nn.Module):
    """Runs LSTMCell over a full sequence"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x, h, s):
        """
        x: (T, batch, input_size)
        h: (batch, hidden_size)
        s: (batch, hidden_size)
        """
        outputs = []

        for t in range(x.shape[0]):
            h, s = self.cell(x[t], h, s)
            outputs.append(h.unsqueeze(0))

        return torch.cat(outputs, dim=0), (h, s)


class LSTM(nn.Module):
    """2-layer LSTM + linear output"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Two stacked LSTM layers
        self.layer1 = LSTMLayer(input_size, hidden_size)
        self.layer2 = LSTMLayer(hidden_size, hidden_size)

        # Linear output (logits, softmax handled by loss)
        self.linear = nn.Linear(hidden_size, output_size)

    @property
    def initial_states(self):
        h = torch.zeros(2, 1, self.hidden_size)
        s = torch.zeros(2, 1, self.hidden_size)
        return h, s

    def forward(self, x, states):
        """
        x:      (T, batch, input_size)
        states: (h, s) each (2, batch, hidden_size)
        """
        h, s = states

        out1, (h1, s1) = self.layer1(x, h[0], s[0])
        out2, (h2, s2) = self.layer2(out1, h[1], s[1])

        h_new = torch.stack([h1, h2], dim=0)
        s_new = torch.stack([s1, s2], dim=0)

        output = self.linear(out2)

        return output, (h_new, s_new)
