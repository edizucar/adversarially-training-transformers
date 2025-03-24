# config.py
from dataclasses import dataclass, field
from typing import Optional, Tuple
import yaml
import os
import torch

@dataclass
class TrainingConfig:
    # I/O
    source_dir: str = 'checkpoints/temp'
    dest_dir: str = 'checkpoints/temp'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    never_save_checkpoint: bool = False
    init_from: str = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
    huggingface_model_id: str = 'roneneldan/TinyStories-33M'
    
    # wandb logging
    wandb_log: bool = False
    wandb_project: str = 'owt'
    wandb_run_name: str = 'gpt2'
    
    # data
    dataset: str = 'tiny_stories'
    gradient_accumulation_steps: int = 40
    batch_size: int = 12
    block_size: int = 1024
    hf_config_block_size: int = 2048
    
    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    bias: bool = True
    
    # optimizer
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # learning rate schedule
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    
    # probe settings
    lambda_adversarial: float = 1e-3
    probe_learning_rate: float = 1e-3
    probe_type: str = "linear"
    phi_probe_steps_per_model_update: int = 1  # steps to train probe per model update
    
    # training components
    train_model: bool = True
    train_probes: bool = True
    train_adversarially: bool = True
    
    # system
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'cuda:0'
    compile: bool = True
    dtype: str = 'auto'  # 'auto', 'float32', 'bfloat16', 'float16'
    backend: str = 'nccl'  # for DDP
    use_gradient_checkpointing: bool = False  # Set to True to enable gradient checkpointing
    auto_tune_batch_size: bool = False

    # Distributed training attributes (with defaults)
    is_distributed: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    master_process: bool = True
    seed_offset: int = 0

    # Debugging
    debug_data: bool = False
    show_final_eval_on_stop: bool = False
    def __post_init__(self):
        """Adjust settings and derive values after initialization"""
        # Auto-detect device if needed
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Auto-detect dtype if needed
        if self.dtype == 'auto':
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = 'bfloat16'
            elif torch.cuda.is_available():
                self.dtype = 'float16'
            else:
                self.dtype = 'float32'
                
        # Set compile to False if on CPU
        if self.device == 'cpu':
            self.compile = False
            
        # Adjust learning rates based on what we're training
        if not self.train_model:
            self.learning_rate = 0
            
        if not self.train_probes:
            self.probe_learning_rate = 0
            
        if not self.train_adversarially:
            self.lambda_adversarial = 0

        # Convert numeric values that might be strings
        self.learning_rate = float(self.learning_rate)
        self.probe_learning_rate = float(self.probe_learning_rate)
        self.lambda_adversarial = float(self.lambda_adversarial)
        self.max_iters = int(self.max_iters)
        self.lr_decay_iters = int(self.lr_decay_iters)
        self.min_lr = float(self.min_lr)
    
    @classmethod
    def from_yaml(cls, yaml_file):
        """Load configuration from a YAML file"""
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"Config file not found: {yaml_file}")
            
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict)
    
    @classmethod
    def from_py(cls, py_file):
        """Load configuration from a Python file in a safe way"""
        # Create a new namespace
        namespace = {}
        
        # Execute the Python file in the namespace
        with open(py_file, 'r') as f:
            code = compile(f.read(), py_file, 'exec')
            exec(code, namespace)
            
        # Filter out internal Python stuff
        config_dict = {k: v for k, v in namespace.items() 
                      if not k.startswith('__') and not callable(v)}
            
        return cls(**config_dict)