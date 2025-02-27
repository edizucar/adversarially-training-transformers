# How to Run

```bash
python train.py config/<config>.yaml
```

## Configuration Files

### `tiny_stories_adv.yaml`
This is the default configuration for adversarial training using linear probes on the TinyStories dataset. The configuration includes:

#### Output and Checkpointing
- `source_dir`: Source of the initial checkpoint ('checkpoints/backup-checkpoint-24-02')
- `dest_dir`: Destination directory for saving new checkpoints ('checkpoints/tiny_stories_adv')
- `max_iters`: Maximum number of training iterations
- `eval_interval`: Frequency of evaluation during training
- `eval_iters`: Number of iterations for each evaluation
- `log_interval`: Frequency of logging during training

#### Dataset Configuration
- `dataset`: Training dataset ('tiny_stories')
- `gradient_accumulation_steps`: Number of steps to accumulate gradients
- `batch_size`: Number of samples per batch
- `block_size`: Context length for sequence modeling

#### Model Architecture
- `n_layer`: Number of transformer layers
- `n_head`: Number of attention heads
- `n_embd`: Embedding dimension
- `dropout`: Dropout probability

#### Optimizer Settings
- `learning_rate`: Base learning rate
- `decay_lr`: Whether to use learning rate decay
- `lr_decay_iters`: Number of iterations over which to decay learning rate
- `min_lr`: Minimum learning rate after decay
- `beta2`: Adam optimizer beta2 parameter
- `warmup_iters`: Number of iterations for learning rate warmup

#### Training Components
- `train_probes`: Whether to train probes
- `train_model`: Whether to train the main model

#### Probe Configuration
- `probe_type`: Type of probe ('linear', 'nonlinear')
- `probe_learning_rate`: Learning rate for probe training
- `lambda_adversarial`: Weight for adversarial loss term
- `train_adversarially`: Whether to use adversarial training

#### Initialization
- `init_from`: How to initialize the model ('resume', 'scratch', 'gpt2', 'gpt2_small', etc.)

### `old.yaml`
Contains the configuration from the old `train_copy.py` script.

### `py_configs`
Directory containing old Python configs used to run the old `train_copy.py` script.