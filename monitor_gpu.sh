#!/bin/bash

# Function to clean up and exit
cleanup() {
    echo -e "\nExiting GPU monitor..."
    exit 0
}

# Trap Ctrl+C (SIGINT)
trap cleanup SIGINT

while true; do
    # Print headers
    echo "gpu_util [%], mem_util [%], mem_used [MiB], mem_total [MiB]"
    
    # Run nvidia-smi for 10 iterations (10 seconds with -l 1)
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 | head -n 10
done