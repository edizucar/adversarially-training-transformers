# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'adversarial-500-model-300x-probe'
special_out_dir = 'delete_me'
end_itr = 4_500
eval_interval = end_itr # keep frequent because we'll overfit
eval_iters = 50
log_interval = 250 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False
never_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'tiny-stories-train'



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
max_iters = end_itr
lr_decay_iters = end_itr # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# Choose what we want to train!
train_probes = True
train_model = False

# probe configuration
probe_type="linear" # "linear" or "nonlinear"
probe_learning_rate = 1e-3
lambda_adversarial = 1e-1
train_adversarially = True


# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
init_from = 'resume'

from datetime import datetime
wandb_run_name = 'mini-gpt-' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f'{"not" if not train_adversarially else ""}' + "-adversarial" + f"-{probe_type}"