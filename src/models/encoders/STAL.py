import torch

class RepeatLayer(torch.nn.Module):
    """
    Copies neurons to match the target size.
    The source size must be a divisor of the target size.
    """
    def __init__(self, source_size, target_size):
        super(RepeatLayer, self).__init__()
        assert target_size > source_size, f"target_size must be greater than source_size: source={source_size}, target={target_size}."
        assert target_size % source_size == 0, f"target_size must be a multiple of source_size: source={source_size}, target={target_size}."

        self.repeat_param = target_size // source_size

    def forward(self, x):
        return x.repeat_interleave(self.repeat_param, dim=1)


class Block(torch.nn.Module):
    def __init__(self, input_size, in_features, out_features, drop_p, name):
        super().__init__()
        self.lin = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(drop_p)
        self.bn = torch.nn.BatchNorm1d(out_features)

        if out_features == input_size:
            self.match = torch.nn.Identity()
        elif out_features > input_size:
            if out_features % input_size != 0:
                raise ValueError(f"{name}: out_features must be a multiple of in_features: in_features={in_features}, out_features={out_features}.")
            pool_size = out_features // input_size
            self.match = torch.nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
        elif out_features < input_size:
            if input_size % out_features != 0:
                raise ValueError(f"{name}: in_features must be a multiple of out_features: in_features={in_features}, out_features={out_features}.")
            self.match = RepeatLayer(out_features, input_size)

    def forward(self, x):
        x = self.lin(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        Z = self.match(x)
        return Z, x
    

class STAL(torch.nn.Module):
    """ The Spike Threshold Adaptive Learning (STAL) implementation! """
    def __init__(self,
                 window_size:int,
                 n_spikes_per_timestep:int,
                 n_channels:int,
                 l1_sz:int,
                 l2_sz:int,
                 drop_p:float):
        super().__init__()

        self.window_size = window_size
        self.n_spikes_per_timestep = n_spikes_per_timestep
        self.n_channels = n_channels
        self.l1_sz = l1_sz
        self.l2_sz = l2_sz
        self.drop_p = drop_p

        input_size = window_size * n_channels
        output_size = input_size * n_spikes_per_timestep

        self.output_size = output_size

        self.Z1 = Block(input_size, input_size, self.l1_sz, drop_p, "Z1")
        self.Z2 = Block(input_size, self.l1_sz, self.l2_sz, drop_p, "Z2")

        self.match_out = RepeatLayer(self.l2_sz, output_size)

        # Learnable threshold parameters
        # Initialzed in the middle such that no large updates are needed (e.g. 0.99 -> 0.01)
        self.threshold_adder = torch.nn.Parameter(torch.Tensor(output_size).uniform_(0.4, 0.6), requires_grad=True)

    def forward(self, x):

        B = x.shape[0]
        
        x = x.view(B, -1) # flatten the input

        z1, x = self.Z1(x)
        z2, x = self.Z2(x)

        x = self.match_out(x)

        # Final surrogate thresholding
        extracted_feats = x

        # Binary thesholds break the gradient propagation, so they cannot be used,
        # therefore we use a surrogate: sigmoid w/ slope=25
        alpha = 25.0
        thresholded_feats = torch.sigmoid(alpha * (extracted_feats - self.threshold_adder.unsqueeze(0)))

        # Clamp the thresholds to avoid numerical instability
        with torch.no_grad():
            self.threshold_adder.clamp_(0.001, 1.0)

        return thresholded_feats, z1, z2

    def print_learnable_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total Learnable Parameters (encoder):", total_params)
