from lstm_cell import LSTMCell
from basic_rnn_cell import BasicRNNCell
from torch import nn, zeros, empty_like


class CustomRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers=1, rnn_type='basic_rnn'):
        """
        Creates an recurrent neural network of type {basic_rnn, lstm_rnn}

        basic_rnn is an rnn whose layers implement a tanH activation function
        lstm_rnn is ann rnn whose layers implement an LSTM cell

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in each layer of the RNN.
        num_layers: (int), the number of RNN layers at each time step
        rnn_type: (string), the desired rnn type. rnn_type is a member of {'basic_rnn', 'lstm_rnn'}
        """
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # create a ModuleList self.rnn to hold the layers of the RNN
        # and append the appropriate RNN layers to it
        self.rnn = nn.ModuleList()


        for i in range(num_layers):
            if rnn_type == 'basic_rnn':
                if i == 0: 
                    self.rnn.append(BasicRNNCell(vocab_size=vocab_size, hidden_size=hidden_size))
                else:
                    self.rnn.append(BasicRNNCell(vocab_size=hidden_size, hidden_size=hidden_size))

                
            
            else:
                if i == 0: 
                    self.rnn.append(LSTMCell(vocab_size=vocab_size, hidden_size=hidden_size))
                else:
                    self.rnn.append(LSTMCell(vocab_size=hidden_size, hidden_size=hidden_size))



    def forward(self, x, h, c):
        """
        Defines the forward propagation of an RNN for a given sequence

        Arguments
        ----------
        x: (Tensor) of size (B x T x n) where B is the mini-batch size, T is the sequence length and n is the
            number of input features. x the mini-batch of input sequence
        h: (Tensor) of size (l x B x m) where l is the number of layers and m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (l x B x m). c is the cell state of the previous time step if the rnn is an LSTM RNN

        Return
        ------
        outs: (Tensor) of size (B x T x m), the final hidden state of each time step in order
        h: (Tensor) of size (l x B x m), the hidden state of the last time step
        c: (Tensor) of size (l x B x m), the cell state of the last time step, if the rnn is a basic_rnn, c should be
            the cell state passed in as input.
        """

        # compute the hidden states and cell states (for an lstm_rnn) for each mini-batch in the sequence

        # outs = zeros(x.shape[0], x.shape[1], h.shape[2])
        # # nxt = zeros(x.shape[0], x.shape[1], h.shape[2])

        # # Go up layers
        # for i in range(self.num_layers):

        #     h_l = h.clone()
        #     c_l = c.clone()
        #     nxt = outs.clone()

        #     if i == 0:
        #         x_t = x.clone()
        #     else:
        #         x_t = outs.clone()
            
        #     # Go across time-steps
        #     for j in range(x.shape[1]):

        #         x_temp = x_t[:,j,:]

        #         # Call forward of current cell and get h to use on next cell in time series
        #         if self.rnn_type == 'basic_rnn':
        #             h_l[i] = self.rnn[i].forward(x_temp, h_l[i])
                
        #         # Call forward of current cell and get h and c to use on next cell in time series
        #         else:
        #             h_l[i], c_l[i] = self.rnn[i].forward(x_temp, h_l[i], c_l[i])
                
        #         # Store the new values to input to the next layer
        #         nxt[:,j,:] = h_l[i].clone()
                
            
        #     #
        #     h = h_l
        #     c = c_l
        #     outs = nxt

        #############################################################################################
        #############################################################################################

        # Create a tensor for outs
        outs = zeros(x.shape[0], x.shape[1], h.shape[2], device=x.device)


        # For every time-step
        for i in range(x.shape[1]):

            # Temporary variables for h and c
            h_l = zeros(h.shape[0], h.shape[1], h.shape[2], device=h.device)
            c_l = zeros(c.shape[0], c.shape[1], c.shape[2], device=c.device)

            # Temporary variable for x for current time-step
            x_temp = x[:,i,:]

            # For every layer
            for j in range(self.num_layers):

                # Call forward of current cell and get h to use on next cell in layer
                if self.rnn_type == 'basic_rnn':
                    output = self.rnn[j].forward(x_temp, h[j])
                
                # Call forward of current cell and get h and c to use on next cell in layer
                else:
                    output, new_c = self.rnn[j].forward(x_temp, h[j], c[j])
                    c_l[j] = new_c
                
                # Update the temporary h for the current layer and temporary x
                h_l[j] = output
                x_temp = output
            
            # Update h and c after a complete time-step
            h = h_l
            c = c_l

            # Store the final layer output after a complete time-step
            outs[:,i,:] = h_l[-1]

        # print(x.device)
        # print(h.device)
        # print(c.device)


        return outs, h, c
