import torch
import torch.nn as nn
import numpy as np
import config
import torch.nn.functional as F

args = config.get_args()
Vth = args.Vth
gamma = args.gamma


class SpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_input,beta=1.0):
        ctx.save_for_backward(x_input)
        ctx.beta = beta
        output = (x_input >= Vth).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_input, = ctx.saved_tensors 
        grad_input = grad_output.clone()
         ## derivative of arctan (scaled)
        grad_input = grad_input * 1 / (1 + gamma * (x_input - Vth)**2)
        return grad_input

class RecurrentReadout(nn.Module):
    """
    A "readout" layer that integrates its own state over time:
       r_t = alpha * r_{t-1} + W_out x_t + b.
    """
    def __init__(self, input_size, output_size, alpha_init=0.9):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Linear mapping from input to output dimension
        self.W_out = nn.Linear(input_size, output_size)
        
        # A learnable 'leak' (alpha)
        self.alpha = nn.Parameter(torch.full((self.output_size,), alpha_init))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.eye_(self.W_out.weight)  # fill 'layer.weight' with identity
            self.W_out.weight.mul_(0.5)
            nn.init.constant_(self.W_out.bias, 0.0)


    def forward(self, x, r0=None):
        
        batch_size, seq_len, C, H, W = x.shape
        if r0 is None:
            r = x.new_zeros(batch_size, self.output_size)
        else:
            r = r0
        
        r_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            # Pool over the spatial dimensions so that x_t becomes [B, C]
            x_t = x_t.mean(dim=[2, 3])
            r = self.alpha * r + self.W_out(x_t)
            
            r_seq.append(r.unsqueeze(1))
        
        r_seq = torch.cat(r_seq, dim=1)  # (batch_size, seq_len, output_size)
        return r_seq, r
    

class GRUlayer(nn.Module):
    """ 
    Spiking ConvGRU layer that uses convolutions (instead of linear layers) 
    to compute the update gate (tempZ) and the candidate membrane potential.
    
    Assumes the input is of shape (B, T, input_channels, height, width).
    """
    def __init__(self, input_channels, hidden_channels, 
                 kernel_size=3, height=8, width=8,
                 alpha_init=0.9):
        super(GRUlayer, self).__init__()
        self.hidden_channels = hidden_channels
        self.height = height
        self.width = width

        self.conv_z = nn.Conv2d(hidden_channels, hidden_channels,
                                kernel_size, padding=kernel_size//2, bias=True)
        self.conv_uz = nn.Conv2d(hidden_channels, hidden_channels,
                                 kernel_size, padding=kernel_size//2, bias=False)

        self.conv_r = nn.Conv2d(input_channels, hidden_channels,
                                kernel_size, padding=kernel_size//2, bias=True)
        self.conv_ur = nn.Conv2d(hidden_channels, hidden_channels,
                                kernel_size, padding=kernel_size//2, bias=False)

        self.conv_i = nn.Conv2d(input_channels, hidden_channels,
                                kernel_size=3,padding=kernel_size//2, bias=True)
        self.conv_ui = nn.Conv2d(hidden_channels, hidden_channels,
                                 kernel_size=3,padding=kernel_size//2,  bias=False)

        self.clamp()

        # The spiking activation 
        self.spikeact = SpikeAct.apply

        # Weights initialization
        k_ff = np.sqrt(1. / hidden_channels)
        k_rec = np.sqrt(1. / hidden_channels)
        nn.init.uniform_(self.conv_z.weight, a=-k_ff, b=k_ff)
        nn.init.uniform_(self.conv_i.weight, a=-k_ff, b=k_ff)
        nn.init.uniform_(self.conv_uz.weight, a=-k_rec, b=k_rec)
        nn.init.uniform_(self.conv_ui.weight, a=-k_rec, b=k_rec)
        if self.conv_z.bias is not None:
            nn.init.uniform_(self.conv_z.bias, a=-k_ff, b=k_ff)
        if self.conv_i.bias is not None:
            nn.init.uniform_(self.conv_i.bias, a=-k_ff, b=k_ff)
        if self.conv_r.bias is not None:
            nn.init.uniform_(self.conv_r.bias, a=-k_ff, b=k_ff)
    def forward(self, x):
        """
        x: input tensor of shape (B, T, input_channels, height, width)
        """
        T = x.size(1)
        B = x.size(0)
        device = x.device

        # Initialize outputs and temporary states as feature maps.
        outputs = torch.zeros((B, T, self.hidden_channels, self.height, self.width), device=device)
        temps = torch.zeros((B, T, self.hidden_channels, self.height, self.width), device=device)
        output_prev = torch.zeros((B, self.hidden_channels, self.height, self.width), device=device)
        temp = torch.zeros_like(output_prev)
        tempcurrent = torch.zeros_like(output_prev)
        

        for t in range(T):
            # x_t is of shape (B, input_channels, height, width)
            x_t = x[:, t, ...]

            tempR = torch.sigmoid(self.conv_r(x_t) + self.conv_ur(output_prev))
            tempcurrent = tempR * tempcurrent + self.conv_i(x_t) + self.conv_ui(output_prev)
            tempZ = torch.sigmoid(self.conv_z(tempcurrent) + self.conv_uz(output_prev)) 
            
            # Update the membrane potential.
            temp = tempZ * temp + (1 - tempZ) * tempcurrent - Vth * output_prev
            
            # Compute the output spike (using the spiking activation function).
            output_prev = self.spikeact(temp)
            
            outputs[:, t, ...] = output_prev
            temps[:, t, ...] = temp

        return temps

    def clamp(self):
        with torch.no_grad():
            self.alpha.data.clamp_(0., 1.)

class Net(nn.Module):
    def __init__(self,num_inputs, hidden_size, num_outputs,height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.num_inputs = num_inputs
        if args.dataset != 'shd':
            self.gru_inputs = num_inputs

        self.num_inputs = num_inputs
        self.hidden_size = hidden_size 
        self.num_outputs = num_outputs
        if args.dataset == 'shd':
            self.scale = 2
        else:
            self.scale = 1
        self.conv_down = nn.Conv3d(in_channels=self.num_inputs, out_channels=self.gru_inputs,kernel_size=(3, 3, 3),
                 stride=(2, 2, 2), padding=(1, 1, 1) )
        self.max_down = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        self.sgru = GRUlayer(input_channels=self.gru_inputs, hidden_channels=self.hidden_size,
                 kernel_size=3, height=self.height//self.scale, width=self.width//self.scale,
                 alpha_init=0.9)
        self.readout = RecurrentReadout(self.hidden_size, self.num_outputs)

    def forward(self, x):
        if args.dataset == 'shd':
            x = x.permute(0, 2, 1, 3, 4)
            x = self.conv_down(x)
            x = x.permute(0, 2, 1, 3, 4)
        if args.dataset == 'DVSGesture':
            x = x.permute(0, 2, 1, 3, 4)
            x = self.max_down(x)
            x = x.permute(0, 2, 1, 3, 4)
        
        x = self.sgru(x)
        outputs,_ = self.readout(x)
       
        return outputs
        
    
