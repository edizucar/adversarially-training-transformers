# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-tiny-stories-new'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 50
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False
never_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'tiny-stories-train'
wandb_run_name = 'mini-gpt'

dataset = 'tiny_stories'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 #1e-3 # with baby networks can afford to go a bit higher
decay_lr = True
max_iters = 10_000_000
lr_decay_iters = 10_000_000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# Choose what we want to train!
train_probes = True
train_model = True

# probe configuration
probe_type="nonlinear" # "linear" or "nonlinear"
probe_learning_rate = 1e-3
lambda_adversarial = 1e-3
train_adversarially = False


# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
init_from = 'resume'
