import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils.rnn import PackedSequence
import os


# V2Implementation: This implementation of LoRALSTM uses nn.LSTM as a super class
class LoRALSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers, subspace_size=2, use_lora=False):
        super(LoRALSTM, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.subspace_size = subspace_size
        self.use_lora = use_lora
        self._lora_weights = []
        self._lora_weights_names = []
        # If use_lora == False it is the same as PyTorch's LSTM
        if self.use_lora:
            for param_name, param in zip(self._flat_weights_names, self._flat_weights):
                # print("START", param_name)
                if "weight" in param_name:
                    _, layer_idx = param_name.split("l")
                    layer_idx = int(layer_idx)

                    # only W_ii, W_if, W_ig, W_io in the first layer have the dimension of hidden_size x input_size
                    # Other weights have the dimension of hidden_size x hidden_size
                    ih_input_size = hidden_size
                    if "_ih" in param_name:
                        suffix = "i" # For weight matrix name
                        add_idx = 0
                        if layer_idx == 0:
                            ih_input_size = input_size
                    else:
                        suffix = "h" # For weight matrix name
                        add_idx = 1


                    # This part defines the BA for all gates (input, forget, cell, and output gates)
                    gate_name_dict = {0:"i", 1:"f", 2:"g", 3:"o"}
                    BA_temp = []
                    for gate_idx, gate_name in gate_name_dict.items():
                        aux_name = f"{suffix}{gate_name}_l{str(layer_idx)}"
                        # print(aux_name)
                        setattr(self, f"A_{aux_name}", nn.Parameter(torch.Tensor(subspace_size, ih_input_size)))
                        setattr(self, f"B_{aux_name}", nn.Parameter(torch.Tensor(hidden_size, subspace_size)))
                        self._lora_weights.append(getattr(self,f"A_{aux_name}"))
                        self._lora_weights_names.append(f"A_{aux_name}")
                        self._lora_weights.append(getattr(self,f"B_{aux_name}"))
                        self._lora_weights_names.append(f"B_{aux_name}")

                        
                        nn.init.normal_(getattr(self, f"A_{aux_name}").data)
                        nn.init.zeros_(getattr(self, f"B_{aux_name}").data)
                        

                        BA_temp.append(getattr(self, f"B_{aux_name}") @ getattr(self, f"A_{aux_name}"))

                    BA_temp = torch.cat(BA_temp) # torch.Size([num_layers, 4, hidden_size, input_size])
                    param.requires_grad = False
                    param_no_grad = param.detach()
                    param_no_grad.requires_grad = False
                    new_val = param_no_grad + BA_temp
                    self._flat_weights[add_idx + (layer_idx*4)] = new_val
