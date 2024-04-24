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
