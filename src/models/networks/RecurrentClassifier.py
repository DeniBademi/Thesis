# Retrieved from https://github.com/freek1/emopain-stl
import torch
import snntorch as snn

class RecurrentClassifier(torch.nn.Module):
    """ Spiking Recurrent Neural Network (SRNN) implementation. 
    
    Note: some commented code was used for testing of a more complex 2 hidden-layer implementation.
    This is not used in the final paper. """
    def __init__(self, window_size: int, n_spikes_per_timestep: int, n_channels: int, lif_beta: float, l1_sz: int, n_classes: int, num_steps: int):
        super().__init__()

        print("Using Recurrent Classifier: Recurrent LIF Neurons.")

        self.num_steps = num_steps
        self.window_size = window_size
        self.n_spikes_per_timestep = n_spikes_per_timestep
        self.n_channels = n_channels
        
        self.fc1 = torch.nn.Linear(window_size * n_channels, l1_sz)
        self.rlif1 = snn.RLeaky(beta=lif_beta, linear_features=l1_sz, V=1)
        
        # Code for adapting to 2 hidden layers (not used in the paper):
        # self.l2_sz = l2_sz
        # if l2_sz > 0:
        #     self.fc2 = torch.nn.Linear(l1_sz, l2_sz)
        #     self.rlif2 = snn.RLeaky(beta=lif_beta, linear_features=l2_sz)
        #     self.fc3 = torch.nn.Linear(l2_sz, n_classes)
        #     self.rlif3 = snn.RLeaky(beta=lif_beta, linear_features=n_classes, V=1)
        # else:
        
        self.fc2 = torch.nn.Linear(l1_sz, n_classes)
        self.rlif2 = snn.RLeaky(beta=lif_beta, linear_features=n_classes, V=1)

    def forward(self, x):
        spk1, mem1 = self.rlif1.init_rleaky()
        spk_rec, mem_rec = [], []

        # Code for adapting to 2 hidden layers (not used in the paper):
        # if self.l2_sz > 0:
        #     spk2, mem2 = self.rlif2.init_rleaky()
        #     spk3, mem3 = self.rlif3.init_rleaky()
        #     for _ in range(self.num_steps):
        #         cur1 = self.fc1(x)
        #         spk1, mem1 = self.rlif1(cur1, spk1, mem1)
        #         cur2 = self.fc2(spk1)
        #         spk2, mem2 = self.rlif2(cur2, spk2, mem2)
        #         cur3 = self.fc3(spk2)
        #         spk3, mem3 = self.rlif3(cur3, spk3, mem3)
        #         spk_rec.append(spk3)
        #         mem_rec.append(mem3)
        # else:        
    
        x = x.reshape(x.size(0), -1, self.window_size * self.n_channels) #
        
        spk2, mem2 = self.rlif2.init_rleaky()
        for i in range(self.n_spikes_per_timestep):
            cur1 = self.fc1(x[:, i]) # give the input per spike timestep
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.rlif2(cur2, spk2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
    


from spikingjelly.clock_driven.neuron import LIFNode, MultiStepLIFNode
import torch.nn as nn
class RecurrentClassifierSJ(nn.Module):
    """
    Spiking Recurrent Neural Network using SpikingJelly
    """
    def __init__(
        self,
        window_size: int,
        n_spikes_per_timestep: int,
        n_channels: int,
        lif_tau: float,
        l1_sz: int,
        n_classes: int,
        num_steps: int
    ):
        super().__init__()
        print("Using SpikingJelly Recurrent Classifier")

        self.num_steps = num_steps
        self.window_size = window_size
        self.n_spikes_per_timestep = n_spikes_per_timestep
        self.n_channels = n_channels

        # First linear layer + LIF neuron
        self.fc1 = nn.Linear(window_size * n_channels, l1_sz)
        # tau parameter roughly inverse of leak (here using beta as time constant)
        self.lif1 = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend='torch')

        # Output layer + LIF neuron
        self.fc2 = nn.Linear(l1_sz, n_classes)
        self.lif2 = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend='torch')

    def forward(self, x: torch.Tensor):
        # x: [batch, spikes, window_size, n_channels] or [batch, n_spikes, ...]
        # Flatten spatial dims
        # Expecting input shape [batch, n_spikes, window_size, n_channels]
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_spikes_per_timestep, -1)

        # Reset membrane states at start of sequence
        self.lif1.reset()
        self.lif2.reset()

        spk_rec = []
        mem_rec = []  # if you also want to record membrane potentials

        for t in range(self.n_spikes_per_timestep):
            # Linear transform
            cur1 = self.fc1(x[:, t])  # [batch, l1_sz]
            # LIF spike
            spk1 = self.lif1(cur1)

            # Output layer
            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)

            spk_rec.append(spk2)
            # optionally record membrane: self.lif2.v
            mem_rec.append(self.lif2.v.clone())

        # Stack over time: shape [time, batch, n_classes]
        spk_tensor = torch.stack(spk_rec, dim=0)
        mem_tensor = torch.stack(mem_rec, dim=0)
        
        
        return spk_tensor.mean(dim=0) #spk_tensor, mem_tensor
