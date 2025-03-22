# Virtual Environment Stuff
To activate, got to marsGPT directory and run:
`source .venv/bin/activate`

To deactivate, run:
`deactivate`


# Tmux cheatsheet
Create a session
`tmux new-session -s <session-name>`

Detach the session:
`Ctrl+b, d`

Re attach the session
`tmux attach-session -t <session-name>`

Logging in to WandB:
`wandb login` or `wandb login --relogin`
then paste your api key and hit enter. api key can be found on wandb website.


# Georges random notes
Run `python train.py config/train_tiny_stories_adv.py` to adversarially train via linear probes on the tiny stories dataset.

Go into `train_tiny_stories_adv.py` to configure:
- out_dir (where the new model checkpoint will be saved)
- special_out_dir (where the model will be loaded in from)

- end_itr (number of model iterations)
- gradient_accumulation_steps (how many probe updates to do per model update: reffered to as phi in the paper)
- lambda_adversarial (the weight of the adversarial loss: reffered to as lambda in the paper)