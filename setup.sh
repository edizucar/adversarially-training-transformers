#!/bin/bash

# Step 1: Install build-essential
echo "Installing build-essential..."
sudo apt update
sudo apt install -y build-essential

# Step 2: Generate RSA keypair and print the public key
echo "Generating RSA keypair..."
ssh-keygen -t rsa -b 4096 -C "your_email@example.com" -f ~/.ssh/id_rsa -N ""
echo "Public key:"
cat ~/.ssh/id_rsa.pub

# Step 3: Create the /mars directory
echo "Creating /mars directory..."
sudo mkdir -p /mars
sudo chown $USER:$USER /mars

# Step 4: Navigate to /mars
cd /mars

# Step 5: Git clone the repository
echo "Cloning the repository..."
git clone git@github.com:edizucar/adversarially-training-transformers.git

# Step 5.5: Navigate into the cloned repository
cd adversarially-training-transformers

# Step 6: Create TinyStories_all_data directory
echo "Creating TinyStories_all_data directory..."
mkdir -p TinyStories_all_data

# Step 7: Download the TinyStories_all_data.tar.gz file
echo "Downloading TinyStories_all_data.tar.gz..."
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz -P TinyStories_all_data

# Step 8: Untar the file and remove the tar file
echo "Extracting TinyStories_all_data.tar.gz..."
tar -xzvf TinyStories_all_data/TinyStories_all_data.tar.gz -C TinyStories_all_data
rm TinyStories_all_data/TinyStories_all_data.tar.gz

# Step 9: Make a Python3 virtual environment and activate it
echo "Creating Python3 virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 9.5: Install necessary Python packages
echo "Installing Python packages..."
pip install torch numpy transformers datasets tiktoken wandb tqdm jaxtyping beartype

# Step 10: Run the prepare.py script from the raw URL
echo "Running prepare.py script..."
wget https://raw.githubusercontent.com/ad8e/TinyStories-cleaner/main/prepare.py -O prepare.py
python3 prepare.py

# Step 12: Run the prepare.py script in data/tiny_stories
echo "Running data/tiny_stories/prepare.py script..."
python3 data/tiny_stories/prepare.py

echo "Script completed successfully."
