import torch
from torch import Tensor, nn
import wandb
import tiktoken
from jaxtyping import Float, Int, Bool
from typing import Dict, List
import os
import numpy as np
from beartype import beartype

import torch.nn.functional as F
import wandb
from torch.optim import Optimizer

lambda_adversarial = 0 #0.001
probe_learning_rate = 1e-3

# linear probe setup stuff
class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class NonLinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonLinearProbe, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100,output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class ProbeCluster:
    def __init__(self, number_of_probes, probe_type: str, input_dim: int, output_dim:int):
        """
        number_of_probes: int -> the number of probes, should be 2x number of transfomer blocks

        probe_type: str -> "linear" or "nonlinear"
        """
        self.__number_of_probes = number_of_probes
        self.__probe_type = probe_type
        self.__input_dim = input_dim
        self.__output_dim = output_dim

        if probe_type == "linear":
            self.__probes : List[LinearProbe|NonLinearProbe] = [LinearProbe(input_dim,output_dim) for _ in range(number_of_probes)]
        elif probe_type == "nonlinear":
            self.__probes : List[LinearProbe|NonLinearProbe] = [NonLinearProbe(input_dim, output_dim) for _ in range(number_of_probes)]
        else:
            raise TypeError(f"ProbeCluster: __init__: probe_type should be 'linear' or 'nonlinear', was {probe_type}")

        # TODO: Consider the Question: Could we have one optimiser containing params for all probes?
        self.probe_optimisers : List[Optimizer] = [torch.optim.Adam(probe.parameters(), lr=probe_learning_rate) for probe in self.__probes]

    def load_state_dicts(self, state_dicts: List[Dict]):
        if len(self.__probes) != len(state_dicts):
            raise ValueError(f"Number of state_dicts {len(state_dicts)} not equal to number of probes {len(self.__probes)}")
        else:
            for probe, state_dict in zip(self.__probes, state_dicts):
                probe.load_state_dict(state_dict)

    @staticmethod
    def load_from_checkpoint(checkpoint: Dict):
        num_probes: int = checkpoint["number_of_probes"]
        probe_type: str = checkpoint["probe_type"]
        probe_input_dim: int = checkpoint["probe_input_dim"]
        probe_output_dim: int = checkpoint["probe_output_dim"]
        probe_state_dicts: List[Dict] = checkpoint["probe_state_dicts"]

        # tidy up state_dicts (This is a random bug with the torch.save/load that kaparthy found, fix has essentially been copied.)
        unwanted_prefix = '_orig_mod.'
        for state_dict in probe_state_dicts:
            for k,_ in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)


        probe_cluster = ProbeCluster(num_probes, probe_type, probe_input_dim, probe_output_dim)
        probe_cluster.load_state_dicts(probe_state_dicts)
        return probe_cluster

    def state_dict(self):
        return {
            "number_of_probes": self.__number_of_probes,
            "probe_type": self.__probe_type,
            "probe_input_dim": self.__input_dim,
            "probe_output_dim": self.__output_dim,
            "probe_state_dicts": [probe.state_dict() for probe in self.__probes]
        }
    
    def set_eval_mode(self):
        for probe in self.__probes:
            probe.eval()
    
    def set_train_mode(self):
        for probe in self.__probes:
            probe.train()

    


    def compute_probe_losses(self, activations: List[Float[Tensor, "batch_size seq d_model"]], target: Int[Tensor, "batch_size block_size"]) -> Float[Tensor, "number_of_probes"]:
        assert len(activations) == self.__number_of_probes

        probe_losses = torch.zeros(self.__number_of_probes)
        for i, (activation, probe) in enumerate(zip(activations, self.__probes)):
            logits = probe(activation).squeeze(-1) # TODO: type annotate me!
            loss : float = F.binary_cross_entropy_with_logits(logits, target).item() # Really not sure about these types
            probe_losses[i] = loss
        return probe_losses
    
    @staticmethod
    def log_probe_losses(probe_losses_list: List[Float[Tensor, "number_of_probes"]]) -> None:
        probe_losses : Float[Tensor, "eval_iters number_of_probes"] = torch.stack(probe_losses_list, dim=0)
        averaged_probe_losses : Float[Tensor, "number_of_probes"] = probe_losses.mean(dim=0)

        log_dict = {}
        for i in range(averaged_probe_losses.shape[0]):
            # Add logging values to log_dict
            pass




def compute_probe_targets_from_training_data(training_data: Int[Tensor, "batch_size block_size"]) -> Int[Tensor, "batch_size block_size"]:
    """
    block_size -> the length of the string which is an element
    """
    training_data_token_string = __detokenize(training_data)
    list_probe_targets = __compute_probe_targets_quotes(training_data_token_string)
    probe_targets = torch.Tensor(list_probe_targets).float()
    return probe_targets

data_dir = os.path.join('data', "tiny_stories")
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

# For testing `detokenize` and `compute_probe_targets`
# block_size = 256
# batch_size = 64
# offsets = torch.randint(len(data) - block_size, (batch_size,))

# batched_training_data : Int[Tensor, "batch_size block_size"] = torch.stack([torch.from_numpy((data[offset:offset+block_size]).astype(np.int64)) for offset in offsets])

def __detokenize(training_data: Int[Tensor, "batch_size block_size"]) -> List[List[str]]:
    """
    returns list of shape (batch_size, block_size)
    """
    enc = tiktoken.get_encoding("gpt2")
    # MARS: biggest token 50146

    results = []
    for i in range(training_data.shape[0]):
        result = [enc.decode([token]) for token in training_data[i]] # type: ignore
        results.append(result)
    return results

# Make another function that combines compute probe targets and detokenize
def __compute_probe_targets_quotes(token_batches : List[List[str]]):
    """
    Takes:
        - tokens_batches : List[str, "batch_size seq"] - a list of tokens (as strings), length batch_size

    Returns
        - all_targets : List[str, "batch_size seq"] - a list that has a 1 if the current token position is in a quote, and a 0 otherwise.

    """

    all_targets = []
    for tokens in token_batches:
        targets = []
        inside_quote = False
        for i, token in enumerate(tokens):
            if i == 0 and token[0] == '"':
                # Question: is it a reasonable assumption that a lack a whitespace after a quotation mark implies you are inside a quote?

                # we don't know if it is a start/end quote so just ignore it
                continue
            if '"' in token:
                if token[0] == '"':
                    # quote is at beginning of token so check previous token
                    is_start_quote = tokens[i-1][-1] in [' ', '\n', '\t']
                else:
                    quote_index = token.index('"')
                    is_start_quote = token[quote_index-1] in [' ', '\n', '\t']
                if is_start_quote:
                    targets.extend([0] * (i - len(targets)))
                else:
                    targets.extend([1] * ((i+1) - len(targets)))
                inside_quote = is_start_quote
        # Handle trailing tokens after the last quote
        targets.extend([int(inside_quote)] * (len(tokens) - len(targets)))
        all_targets.append(targets)
    return all_targets

def compute_in_quote_probe_loss(probe, activation, target, print_stuff=False):
    logits = probe(activation)
    # print(logits.shape)
    logits = logits.squeeze(-1)
    if print_stuff:
        print('=============')
        print(torch.sigmoid(logits[0]))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, target)
    return loss



# Not sure what to do with this
# def __compute_prev_token_probe_loss(probe, activation):
#     probe_input = activation[:, 1:, :].contiguous().view(-1, n_embd)
#     probe_target = activation[:, :-1, :].contiguous().view(-1, n_embd)
#     probe_output = probe(probe_input)
#     probe_loss = torch.nn.functional.mse_loss(probe_output, probe_target)
#     probe_loss /= probe_target.norm()
#     # probe_loss = torch.nn.functional.mse_loss(probe_output, probe_input)
#     return probe_loss