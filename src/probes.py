import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler

class LinearProbe(nn.Module):
    """Simple linear probe for feature detection"""
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class NonLinearProbe(nn.Module):
    """Non-linear probe with one hidden layer for feature detection"""
    def __init__(self, input_dim, output_dim):
        super(NonLinearProbe, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class ProbeCluster:
    """Cluster of probes for model layer analysis"""
    def __init__(self, n_layer, d_model, learning_rate=1e-3, probe_type="linear", output_dim=1, device="cuda"):
        """
        Args:
            n_layer: Number of transformer layers
            d_model: Dimension of model embeddings
            learning_rate: Learning rate for probe optimizers
            probe_type: "linear" or "nonlinear"
            output_dim: Output dimension for probes (typically 1 for binary features)
            device: Device to put probes on
        """
        self.n_layer = n_layer
        self.d_model = d_model
        self.lr = learning_rate
        self.number_of_probes = n_layer * 2  # For both attention and MLP outputs
        self.probe_type = probe_type
        self.output_dim = output_dim
        
        # Create probes based on type
        if probe_type == "linear":
            self.probes = nn.ModuleList([
                LinearProbe(d_model, output_dim) for _ in range(self.number_of_probes)
            ]).to(device)
        elif probe_type == "nonlinear":
            self.probes = nn.ModuleList([
                NonLinearProbe(d_model, output_dim) for _ in range(self.number_of_probes)
            ]).to(device)
        else:
            raise ValueError(f"probe_type should be 'linear' or 'nonlinear', was {probe_type}")
        
        # Create optimizers
        self.optimizers = [
            torch.optim.Adam(probe.parameters(), lr=learning_rate)
            for probe in self.probes
        ]
    
    def load_state_dicts(self, state_dicts):
        """Load state dictionaries for all probes"""
        if len(self.probes) != len(state_dicts):
            raise ValueError(f"Number of state_dicts {len(state_dicts)} not equal to number of probes {len(self.probes)}")
        else:
            for probe, state_dict in zip(self.probes, state_dicts):
                probe.load_state_dict(state_dict)

    @staticmethod
    def load_from_checkpoint(checkpoint, lr, device="cuda"):
        """Create a ProbeCluster from a checkpoint dictionary"""
        num_probes = checkpoint["number_of_probes"]
        probe_type = checkpoint["probe_type"]
        probe_input_dim = checkpoint["probe_input_dim"]
        probe_output_dim = checkpoint["probe_output_dim"]
        probe_state_dicts = checkpoint["probe_state_dicts"]

        # Fix key prefixes if needed
        unwanted_prefix = '_orig_mod.'
        for state_dict in probe_state_dicts:
            for k in list(state_dict.keys()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        n_layer = num_probes // 2  # Assuming 2 probes per layer
        probe_cluster = ProbeCluster(
            n_layer=n_layer, 
            d_model=probe_input_dim, 
            learning_rate=lr, 
            probe_type=probe_type, 
            output_dim=probe_output_dim, 
            device=device
        )
        probe_cluster.load_state_dicts(probe_state_dicts)
        return probe_cluster

    def state_dict(self):
        """Get state dictionary for saving"""
        return {
            "number_of_probes": self.number_of_probes,
            "probe_type": self.probe_type,
            "probe_input_dim": self.d_model,
            "probe_output_dim": self.output_dim,
            "probe_state_dicts": [probe.state_dict() for probe in self.probes]
        }
    
    def set_eval_mode(self):
        """Set all probes to evaluation mode"""
        for probe in self.probes:
            probe.eval()
    
    def set_train_mode(self):
        """Set all probes to training mode"""
        for probe in self.probes:
            probe.train()

    def compute_probe_losses(self, activations, targets):
        """
        Compute BCE loss for each probe
        
        Args:
            activations: List of tensors of shape [batch_size, seq_len, d_model]
            targets: Tensor of shape [batch_size, seq_len]
            
        Returns:
            List of scalar tensors containing loss for each probe
        """
        assert len(activations) == self.number_of_probes, f"Expected {self.number_of_probes} activations, got {len(activations)}"
        
        losses = []
        for act, probe in zip(activations, self.probes):
            logits = probe(act).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, targets.float())
            losses.append(loss)
        return losses

    def compute_probe_predictions(self, activations, threshold=0.5):
        """
        Compute binary predictions for each probe
        
        Args:
            activations: List of tensors of shape [batch_size, seq_len, d_model]
            threshold: Threshold for binary prediction
            
        Returns:
            List of tensors containing binary predictions
        """
        predictions = []
        for act, probe in zip(activations, self.probes):
            logits = probe(act).squeeze(-1)
            prediction = (F.sigmoid(logits) > threshold)
            predictions.append(prediction)
        return predictions

    def compute_accuracies(self, activations, targets, threshold=0.5):
        """
        Compute accuracy for each probe
        
        Args:
            activations: List of tensors of shape [batch_size, seq_len, d_model]
            targets: Tensor of shape [batch_size, seq_len]
            threshold: Threshold for binary prediction
            
        Returns:
            List of floats containing accuracy for each probe
        """
        predictions = self.compute_probe_predictions(activations, threshold)
        accuracies = [(pred == targets).float().mean().item() for pred in predictions]
        return accuracies

    def update_probes(self, activations, targets, scaler=None):
        """
        Update all probes in one go
        
        Args:
            activations: List of tensors of shape [batch_size, seq_len, d_model]
            targets: Tensor of shape [batch_size, seq_len]
            scaler: Optional GradScaler for mixed precision training
        
        Returns:
            List of losses
        """
        # Compute losses
        losses = self.compute_probe_losses(activations, targets)
        
        # Update each probe
        for i, (loss, opt) in enumerate(zip(losses, self.optimizers)):
            opt.zero_grad()
            if scaler is not None:
                if i < len(losses) - 1:
                    scaler.scale(loss).backward(retain_graph=True)
                else:
                    scaler.scale(loss).backward()
                scaler.step(opt)
            else:
                if i < len(losses) - 1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                opt.step()
        
        return losses

    def zero_grad_probes(self):
        """Zero gradients for all probe optimizers"""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)