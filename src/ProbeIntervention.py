import torch
from torch import Tensor, nn
import wandb
import tiktoken
from tiktoken import Encoding as Tokenizer
from jaxtyping import Float, Int, Bool, jaxtyped
from typeguard import typechecked
from typing import Dict, List, NamedTuple, Callable
import os
import numpy as np
from beartype import beartype

import torch.nn.functional as F
import wandb
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler

GPT2_TOKENIZER = tiktoken.get_encoding("gpt2")

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


BinaryFeatureExtractorOutput = NamedTuple(
    "FeatureExtractorOutput",
    [
        ("text", list[list[str]]), # batched (hence nested)
        ("tokens", Int[torch.Tensor, "batch seq_len"]),
        ("features", Int[torch.Tensor, "batch seq_len"]),
    ],
)

FeatureExtractor = Callable[[Int[Tensor, "batch seq_len"], Tokenizer],BinaryFeatureExtractorOutput]

@jaxtyped(typechecker=beartype)
def in_quotes_feature(
        tokens: Int[Tensor, "batch seq_len"],
        tokenizer: Tokenizer = GPT2_TOKENIZER,
        qmark_in_quote: bool = False
) -> BinaryFeatureExtractorOutput:
    batch_size,seq_len = tokens.shape
    text : List[List[str]] = __detokenize(tokens, tokenizer)

    assert len(text) == batch_size and len(text[0]) == seq_len

    quote_features: list[list[bool]] = []

    for batch in text:

        quote_feature: list[bool] = []
        inside_quote: bool = False # assume we arent in a quote to start
        is_qmark: bool = False

        for token in batch:
            if '"' in token:
                is_qmark = True
                inside_quote = not inside_quote
            else:
                is_qmark = False

            quote_feature.append(
                qmark_in_quote # tokens with `"` treated separately 
                if is_qmark 
                else inside_quote # otherwise, use the current state
            )

        quote_features.append(quote_feature)

    return BinaryFeatureExtractorOutput(
        text=text,
        tokens=tokens,
        features=torch.tensor(quote_features, device=tokens.device, dtype=torch.float),
    )

@jaxtyped(typechecker=beartype)
def in_quotes_feature_demian(
        tokens: Int[Tensor, "batch seq_len"],
        tokenizer: Tokenizer = GPT2_TOKENIZER,
        qmark_in_quote: bool = False
) -> BinaryFeatureExtractorOutput:
    batch_size,seq_len = tokens.shape
    texts : List[List[str]] = __detokenize(tokens, tokenizer)

    assert len(texts) == batch_size and len(texts[0]) == seq_len

    quote_features: list[list[bool]] = []

    for text in texts:

        quote_feature: list[bool] = []
        inside_quote: bool = False # assume we arent in a quote to start

        for i, token in enumerate(text):
            if i == 0 and token[0] == '"':
                # we don't know if it is a start/end quote so just ignore it
                continue
            if '"' in token:
                # token[0] == '."'
                if token[0] == '"':
                    # quote is at beginning of token so check previous token
                    is_start_quote = text[i-1][-1] in [' ', '\n', '\t']
                else:
                    quote_index = token.index('"')
                    is_start_quote = token[quote_index-1] in [' ', '\n', '\t']
                if is_start_quote:
                    quote_feature.extend([False] * ((i + int(not qmark_in_quote)) - len(quote_feature)))
                else:
                    quote_feature.extend([True] * ((i + int(qmark_in_quote)) - len(quote_feature)))
                inside_quote = is_start_quote
        # Handle trailing tokens after the last quote
        quote_feature.extend([inside_quote] * (len(text) - len(quote_feature)))
        quote_features.append(quote_feature)

    return BinaryFeatureExtractorOutput(
        text=texts,
        tokens=tokens,
        features=torch.tensor(quote_features, device=tokens.device, dtype=torch.float),
    )

in_quotes_no_qmark_feature: FeatureExtractor = lambda tokens, tokenizer: in_quotes_feature(tokens, tokenizer, qmark_in_quote=False)
in_quotes_with_qmark_feature: FeatureExtractor = lambda tokens, tokenizer: in_quotes_feature(tokens, tokenizer, qmark_in_quote=True)


# tokens: Int64[Tensor, "seq_len"] <- this doesn't work... why???
#@jaxtyped(typechecker=typechecked)
def display_features_from_tokens_and_feature_tensor(tokens : Int[Tensor, "seq_len"] , feature_tensor: Bool[Tensor, "seq_len"]):
    text = __detokenize(tokens.unsqueeze(dim=0))[0]
    # Print Key:
    print(f"------ Probe Feature Prediction --------")
    for  token, feature in zip(text,feature_tensor):
        if feature == 1:
            # print(f"\033[43m{token}\033[0m", end=" ")
            # dark blue background
            print(f"\033[44m'{token}'\033[0m", end=" ")
        else:
            print(f"'{token}'", end=" ")
    print("\n ------------------------------------")


def get_token_str_from_token_idx(tokens: Int[Tensor, "seq_len"]) -> List[str]:
    return __detokenize(tokens.unsqueeze(dim=0))[0]


@jaxtyped(typechecker=beartype)
def display_features_from_tokens(tokens: Int[Tensor, "seq_len"], tokenizer: Tokenizer = GPT2_TOKENIZER, feature_extractor: FeatureExtractor = in_quotes_feature_demian) -> None:
    seq_len = tokens.shape[0]
    text,_, quote_features_tensor = feature_extractor(tokens.unsqueeze(dim=0), tokenizer)
    assert len(text) == 1 and len(text[0]) == seq_len
    display_features_from_tokens_and_feature_tensor(tokens[0], quote_features_tensor.squeeze(dim=0))

@jaxtyped(typechecker=beartype)
def display_features_from_text(text: str, tokenizer: Tokenizer, feature_extractor: FeatureExtractor) -> None:
    tokens : Int[Tensor, "seq_len"] = torch.Tensor(tokenizer.encode(text)).type(torch.int)
    display_features_from_tokens(tokens, tokenizer, feature_extractor)

class ProbeCluster:
    def __init__(self, number_of_probes, probe_type: str, input_dim: int, output_dim:int, lr: float, device:str="cpu"):
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

        for probe in self.__probes:
            probe.to(device)
        # TODO: Consider the Question: Could we have one optimiser containing params for all probes?
        self.__probe_optimisers : List[Optimizer] = [torch.optim.Adam(probe.parameters(), lr=lr) for probe in self.__probes]

    def load_state_dicts(self, state_dicts: List[Dict]):
        if len(self.__probes) != len(state_dicts):
            raise ValueError(f"Number of state_dicts {len(state_dicts)} not equal to number of probes {len(self.__probes)}")
        else:
            for probe, state_dict in zip(self.__probes, state_dicts):
                probe.load_state_dict(state_dict)

    @staticmethod
    def load_from_checkpoint(checkpoint: Dict, lr, device="cpu"):
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


        probe_cluster = ProbeCluster(num_probes, probe_type, probe_input_dim, probe_output_dim, lr=lr, device=device)
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

    def get_num_probes(self) -> Int:
        return self.__number_of_probes

    @jaxtyped(typechecker=beartype)
    def compute_probe_losses(self, activations: List[Float[Tensor, "batch_size seq d_model"]], target: Float[Tensor, "batch_size seq_len"]) -> List[Float[Tensor, ""]]:
        """
        Returns a List of length number_of_probes.
        Each element of the list is a 0 rank Tensor
        which contains a single float representing the loss of that probe.
        """
        
        assert len(activations) == self.__number_of_probes

        probe_losses : List[Float] = []
        for i, (activation, probe) in enumerate(zip(activations, self.__probes)):
            logits : Float[Tensor, "batch_size seq"] = probe(activation).squeeze(-1)
            loss : Float[Tensor, "batch_size seq"] = F.binary_cross_entropy_with_logits(logits, target)
            probe_losses.append(loss)
        return probe_losses

    def compute_probe_predictions(self, activations: List[Float[Tensor, "batch_size seq d_model"]], threshold: Float = .5) -> List[Bool[Tensor, "batch_size seq_len"]]:
        predictions : List[Bool[Tensor, "batch_size seq_len"]] = []
        for (activation, probe) in zip(activations, self.__probes):
            logits : Float[Tensor, "batch_size seq"] = probe(activation).squeeze(-1)
            prediction = F.sigmoid(logits) > threshold
            predictions.append(prediction)
        return predictions

    def compute_accuracies_from_predictions(self, predictions: List[Bool[Tensor, "batch_size seq_len"]], targets: Bool[Tensor, "batch_size seq_len"]) -> List[Float[Tensor, "batch_size seq_len"]]:
        accuracies : List[Float[Tensor, "batch_size seq_len"]] = []
        for prediction in predictions:
            accuracy = prediction == targets
            accuracies.append(accuracy)
        return accuracies
    
    def compute_accuracies(self, activations: List[Float[Tensor, "batch_size seq_len d_model"]], targets: Bool[Tensor, "batch_size seq_len"], threshold : Float = .5) -> List[Float]:
        """
        Returns a list of accuracies, one for each probe
        """
        predictions = self.compute_probe_predictions(activations, threshold)
        accuracies = [probe_acc.type(torch.float).mean().item() for probe_acc in self.compute_accuracies_from_predictions(predictions, targets)]
        return accuracies

    @jaxtyped(typechecker=beartype)
    def display_probe_predictions(self, text: str, model, device, tokenizer: Tokenizer = GPT2_TOKENIZER) -> None:
        tokens : Int[Tensor, "seq_len"] = torch.Tensor(tokenizer.encode(text)).to(device).type(torch.int64)
        batched_tokens : Int[Tensor, "batch_size seq_len"] = tokens.unsqueeze(dim=0)
        activations : List[Float[Tensor, "1 seq_len d_model"]]

        _, _, activations = model(batched_tokens[:,:-1],batched_tokens[:,1:])
        assert len(activations) == self.__number_of_probes

        predictions = self.compute_probe_predictions(activations)
        for i,prediction in enumerate(predictions):
            print(f"Probe {i}:")
            display_features_from_tokens_and_feature_tensor(tokens, prediction.squeeze(dim=0))
            print("\n")

    def loss_backward_probes(self, activations : List[Float[Tensor, "batch_size seq d_model"]], probe_targets, scaler: GradScaler):
        """Optimized version with less CPU-GPU synchronization"""
        self.set_train_mode()
        detached_activations = [activation.detach() for activation in activations]
        
        # Compute all probe losses in one go
        probe_losses = self.compute_probe_losses(detached_activations, probe_targets)
        
        # Scale and backward all losses together if possible
        for i, probe_loss in enumerate(probe_losses):
            if i < len(probe_losses) - 1:
                # For all but the last loss, retain graph
                scaler.scale(probe_loss).backward(retain_graph=True)
            else:
                # For the last loss, no need to retain graph
                scaler.scale(probe_loss).backward()
        
        scaler.update()

    def optimiser_step_probes(self, scaler: GradScaler):
        for probe_optimiser in self.__probe_optimisers:
            scaler.step(probe_optimiser)
        self.zero_grad_probes()

    def zero_grad_probes(self):
        for probe_optimiser in self.__probe_optimisers:
            probe_optimiser.zero_grad(set_to_none=True)

    def show_predictions(self, text: List[str], activations: List[Float[Tensor, "seq_len d_model"]]):
        pass

def compute_probe_targets_from_training_data(training_data: Int[Tensor, "batch_size block_size"]) -> Int[Tensor, "batch_size block_size"]:
    """
    block_size -> the length of the string which is an element
    """
    training_data_token_string = __detokenize(training_data)
    list_probe_targets = __compute_probe_targets_quotes(training_data_token_string)
    probe_targets = torch.Tensor(list_probe_targets).float()
    return probe_targets

#@jaxtyped(typechecker=beartype)
def __detokenize(training_data: Int[Tensor, "batch_size block_size"], tokenizer=GPT2_TOKENIZER) -> List[List[str]]:
    """
    returns list of shape (batch_size, block_size)
    """
    # MARS: biggest token 50146

    results = []
    for i in range(training_data.shape[0]):
        result = [tokenizer.decode([token.item()]) for token in training_data[i]] # type: ignore
        results.append(result)
    return results

# Make another function that combines compute probe targets and detokenize
def __compute_probe_targets_quotes(token_batches : List[List[str]]):
    """
    Takes:
        - tokens_batches : List[str, "batch_size seq"] - a list of tokens (as strings), length batch_size

    Returns
        - all_targets : List[int, "batch_size seq"] - a list that has a 1 if the current token position is in a quote, and a 0 otherwise.

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