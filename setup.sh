pip install -r requirements.txt
pip install flash-attn --no-build-isolation

if [ "$1" = "--get-checkpoint" ]; then
    mkdir -p ./checkpoints/backup-checkpoint-24-0/ && wget -O ./checkpoints/backup-checkpoint-24-02/ckpt.pt https://github.com/edizucar/adversarially-training-transformers/releases/download/pretrain_checkpoint/backup_checkpoint_24_02_ckpt.pt
    mkdir -p ./checkpoints/openwebtext/ && wget -O ./checkpoints/openwebtext/ckpt.pt https://github.com/edizucar/adversarially-training-transformers/releases/download/pretrain_checkpoint/openwebtext_ckpt.pt
    mkdir -p ./checkpoints/tiny_stories/ && wget -O ./checkpoints/tiny_stories/ckpt.pt https://github.com/edizucar/adversarially-training-transformers/releases/download/pretrain_checkpoint/tiny_stories_ckpt.pt
fi