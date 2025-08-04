from torch import nn, sigmoid, tanh, Tensor
from math import sqrt
import torch


class LSTMCell(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        """
        Creates an RNN layer with an LSTM activation function

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in the rnn cell.

        """
        super(LSTMCell, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # create and initialize parameters W, V, b as described in the text.
        # remember that the parameters are instance variables

        self.b = nn.Parameter(Tensor(hidden_size * 4).uniform_(sqrt(1/hidden_size)*-1, sqrt(1/hidden_size)))

        self.w = nn.Parameter(Tensor(vocab_size, hidden_size * 4).uniform_(sqrt(1/hidden_size)*-1, sqrt(1/hidden_size)))

        self.v = nn.Parameter(Tensor(hidden_size, hidden_size * 4).uniform_(sqrt(1/hidden_size)*-1, sqrt(1/hidden_size)))

        # W, the input weights matrix has size (n x (4 * m)) where n is
        # the number of input features and m is the hidden size
        # V, the hidden state weights matrix has size (m, (4 * m))
        # b, the vector of biases has size (4 * m)

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an LSTM layer

        Arguments
        ---------
        x: (Tensor) of size (B x n) where B is the mini-batch size and n is the number of input-features.
            If the RNN has only one layer at each time step, x is the input data of the current time-step.
            In a multi-layer RNN, x is the previous layer's hidden state (usually after applying a dropout)
        h: (Tensor) of size (B x m) where m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (B x m), the cell state of the previous time step

        Return
        ------
        h_out: (Tensor) of size (B x m), the new hidden
        c_out: (Tensor) of size (B x m), he new cell state

        """

        # pre-activation
        a = self.b + Tensor.matmul(x, self.w) + Tensor.matmul(h, self.v)

        i, f, o, g = torch.chunk(a, 4, dim=-1)

        i = sigmoid(i)

        f = sigmoid(f)

        o = sigmoid(o)

        g = tanh(g)

        c_out = (i * g) + (f * c)

        h_out = o * tanh(c_out)

        return h_out, c_out


