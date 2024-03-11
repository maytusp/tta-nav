# Implemented from scratch: deos not work

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils.rnn import PackedSequence
import os

# Define the LoRALSTM cell
class LoRALSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, subspace_size, use_lora=False):
        super(LoRALSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.subspace_size = subspace_size
        self.use_lora = use_lora
        #self.A and self.B are for low-rank projection

        # Input gate
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

        self.A_ii = nn.Parameter(torch.Tensor(subspace_size, input_size))
        self.B_ii = nn.Parameter(torch.Tensor(hidden_size, subspace_size))
        self.A_hi = nn.Parameter(torch.Tensor(subspace_size, hidden_size))
        self.B_hi = nn.Parameter(torch.Tensor(hidden_size, subspace_size))

        # Forget gate
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

        self.A_if = nn.Parameter(torch.Tensor(subspace_size, input_size))
        self.B_if = nn.Parameter(torch.Tensor(hidden_size, subspace_size))
        self.A_hf = nn.Parameter(torch.Tensor(subspace_size, hidden_size))
        self.B_hf = nn.Parameter(torch.Tensor(hidden_size, subspace_size))

        # Cell gate
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))

        self.A_ig = nn.Parameter(torch.Tensor(subspace_size, input_size))
        self.B_ig = nn.Parameter(torch.Tensor(hidden_size, subspace_size))
        self.A_hg = nn.Parameter(torch.Tensor(subspace_size, hidden_size))
        self.B_hg = nn.Parameter(torch.Tensor(hidden_size, subspace_size))

        # Output gate
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))

        self.A_io = nn.Parameter(torch.Tensor(subspace_size, input_size))
        self.B_io = nn.Parameter(torch.Tensor(hidden_size, subspace_size))
        self.A_ho = nn.Parameter(torch.Tensor(subspace_size, hidden_size))
        self.B_ho = nn.Parameter(torch.Tensor(hidden_size, subspace_size))


        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if self.use_lora:
                if "A_" in name:
                    nn.init.normal_(p.data)
                elif "B_" in name:
                    nn.init.zeros_(p.data)
                else:
                    p.requires_grad = False # Disable other parameters to not be trained
            else:
                if p.data.ndimension() >= 2:
                    if "A_" in name or "B_" in name:
                        nn.init.zeros_(p.data)
                        p.requires_grad = False
                    else:
                        nn.init.xavier_uniform_(p.data)
                else:
                    nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        """
        Forward pass of the LSTM cell.
        Args:
            x (torch.Tensor): Input tensor for the current time step.
            init_states (tuple): Tuple of initial hidden and cell states.
        Returns:
            tuple: Tuple containing the updated hidden and cell states.
        """
        bs, input_size = x.size()

        # Initialize hidden and cell states if not provided
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size),
                torch.zeros(bs, self.hidden_size),
            )
        else:
            h_t, c_t = init_states

        # Input gate
        i_t = torch.sigmoid(x @ self.W_ii.t() + self.b_ii + h_t @ self.W_hi.t() + self.b_hi +
                            x @ (self.B_ii @ self.A_ii).t() + h_t @ (self.B_hi @ self.A_hi).t())

        # Forget gate
        f_t = torch.sigmoid(x @ self.W_if.t() + self.b_if + h_t @ self.W_hf.t() + self.b_hf +
                            x @ (self.B_if @ self.A_if).t() + h_t @ (self.B_hf @ self.A_hf).t())
        # Cell gate
        g_t = torch.tanh(x @ self.W_ig.t() + self.b_ig + h_t @ self.W_hg.t() + self.b_hg +
                         x @ (self.B_ig @ self.A_ig).t() + h_t @ (self.B_hg @ self.A_hg).t())

        # Update cell state
        c_t = f_t * c_t + i_t * g_t

        # Output gate
        o_t = torch.sigmoid(x @ self.W_io.t() + self.b_io + h_t @ self.W_ho.t() + self.b_ho +
                            x @ (self.B_io @ self.A_io).t() + h_t @ (self.B_ho @ self.A_ho).t())

        # Update hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


# Define the 2-layer LSTM model
class LoRALSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, subspace_size=16, use_lora=False):
        super(LoRALSTMModel, self).__init__()
        if isinstance(hidden_size, list):
            self.hidden_size = hidden_size
        elif isinstance(hidden_size, int):
            self.hidden_size = [hidden_size for _ in range(num_layers)]
        else:
            assert("ERROR: hidden_size has to be either integer (for multiple layers with identical size) or list")

        self.num_layers = num_layers

        # Stack multiple LSTM layers
        self.lstm_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.lstm_layers.append(LoRALSTMCell(input_size, self.hidden_size[layer], subspace_size, use_lora))
            else:
                self.lstm_layers.append(LoRALSTMCell(self.hidden_size[layer-1], self.hidden_size[layer], subspace_size, use_lora))


    def forward(self, input_seq, init_states):
        h0, c0 = init_states
        h_t, c_t = [None] * self.num_layers, [None] * self.num_layers
        isPackedSequence = isinstance(input_seq, PackedSequence)
        H = h0[-1][0].shape[0]
        if isPackedSequence:
            data, batch_sizes, sorted_indices, unsorted_indices = input_seq
            max_batch_size = batch_sizes[0]
            max_time_steps = len(batch_sizes)
            
            # Process each time step
            init_step = 0
            for step in range(max_time_steps):
                current_batch_size = batch_sizes[step]
                final_step = init_step + current_batch_size
                inp = data[init_step:final_step]
                h0, c0 = h0[:,:current_batch_size,:], c0[:,:current_batch_size,:]
                for layer in range(self.num_layers):
                    h_t[layer], c_t[layer] = self.lstm_layers[layer](inp, (h0[layer], c0[layer]))
                    inp = h_t[layer]
                # Padd h_out (last layer)
                pad_size = max_batch_size - current_batch_size
                h_out = nn.functional.pad(input=h_t[-1], pad=(0,0,0,pad_size), mode='constant', value=0)
                if step == 0:
                    out = []
                out.append(h_out)
        else:
            # Process each time step
            for step, x in enumerate(input_seq):
                inp = x
                for layer in range(self.num_layers):
                    h_t[layer], c_t[layer] = self.lstm_layers[layer](inp, (h0[layer], c0[layer]))
                    inp = h_t[layer]
                h_out = h_t[-1]
                if step == 0:
                    out = []
                out.append(h_out)

        out = torch.stack([x for x in h_out])
        if isPackedSequence:
            out = PackedSequence(out, batch_sizes, sorted_indices, unsorted_indices)
        h_t = torch.stack([x for x in h_t])
        c_t = torch.stack([x for x in c_t])
        # Use the hidden state of the last layer for prediction
        return out, (h_t, c_t)