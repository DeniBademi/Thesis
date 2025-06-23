import torch
import numpy as np

class SpikeSparsity:
    """
    Class for computing spike sparsity metrics from neural network embeddings.
    
    This class provides methods to calculate various sparsity metrics on 
    spike-based embeddings, particularly useful for analyzing efficiency of 
    spiking neural networks.
    """
    
    @staticmethod
    def binary_sparsity(embeddings: torch.Tensor, threshold=0.5):
        """
        Calculate sparsity as the percentage of values below threshold.
        
        Args:
            embeddings: Tensor with shape [T, B, N, C] where:
                T: time steps
                B: batch size
                N: number of spatial positions/patches
                C: embedding channels
            threshold: Threshold value for considering an activation as "off" (default: 0.5)
            
        Returns:
            float: Sparsity ratio (0.0-1.0) where higher value means more sparse
                  (i.e., percentage of "silent" neurons)
        """

        # Count values below threshold
        silent = (embeddings <= threshold).float()
        
        # Calculate ratio of silent activations
        sparsity = torch.mean(silent)
        
        return sparsity
    
    @staticmethod
    def activation_sparsity(embeddings: torch.Tensor):

        # Flatten all dimensions for norm calculation
        flat_embeddings = embeddings.reshape(-1)
        
        # Calculate L1 and L2 norms
        l1_norm = torch.norm(flat_embeddings, p=1)
        l2_norm = torch.norm(flat_embeddings, p=2)
        
        # Prevent division by zero
        if l2_norm == 0:
            return torch.tensor(1.0)
            
        # Calculate sparsity measure: 1 - (L1/L2)/sqrt(n)
        # This equals 1 for a 1-sparse vector and 0 for a dense vector with equal values
        n = flat_embeddings.numel()
        hoyer_sparsity = torch.tensor(1.0) - (l1_norm / (l2_norm * torch.sqrt(torch.tensor(n, dtype=torch.float32))))
        
        return hoyer_sparsity
    
    @staticmethod
    def temporal_sparsity(embeddings: torch.Tensor, threshold=0.5):

 
        # Create binary activation tensor
        binary_acts = (embeddings > threshold).float()
        
        # Sum across time dimension (dim=0) to get activation count per neuron
        active_timesteps = torch.sum(binary_acts, dim=0)
        
        # Average number of time steps each neuron is active
        avg_active_time = torch.mean(active_timesteps)
        
        return avg_active_time
    
    @staticmethod
    def spike_density(embeddings: torch.Tensor):
        """Compute the spike density of the embeddings as described in the STAL paper
        """
        
        spk_inp_test = embeddings
        n_spikes = torch.sum(spk_inp_test)
        total_spikes = torch.tensor(np.prod(spk_inp_test.shape))
        sparsity = n_spikes / total_spikes
        return sparsity
        
    
    @staticmethod
    def compute_all_metrics(embeddings: torch.Tensor, threshold=0.5):
        """
        Compute all sparsity metrics in a single call.
        
        Args:
            embeddings: Tensor with shape [T, B, N, C]
            threshold: Threshold value for binary metrics
            
        Returns:
            dict: Dictionary containing all sparsity metrics
        """
        return {
            "binary_sparsity": SpikeSparsity.binary_sparsity(embeddings, threshold),
            "activation_sparsity": SpikeSparsity.activation_sparsity(embeddings),
            "temporal_sparsity": SpikeSparsity.temporal_sparsity(embeddings, threshold),
            "spike_density": SpikeSparsity.spike_density(embeddings)
        } 
        
