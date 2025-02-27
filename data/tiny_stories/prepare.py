# saves the tinystories dataset to a binary file for training
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# number of workers in .map() call
num_proc = 8
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    # Load TinyStories dataset directly from Hugging Face
    print("Loading TinyStories dataset from Hugging Face")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # Create a validation split
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val
    
    print(split_dataset)
    
    # Tokenize the dataset with GPT-2 BPE
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token (50256 for gpt2 bpe)
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    # Tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    
    # Save metadata (vocab size)
    import pickle
    meta = {'vocab_size': enc.n_vocab}
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    # Concatenate all ids into binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        dtype = np.uint16  # enc.max_token_value == 50256 is < 2**16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    
    print(f"Processing complete. Files saved to data/tiny_stories")
    print(f"Train size: {os.path.getsize('train.bin') / (1024 * 1024):.2f} MB")
    print(f"Val size: {os.path.getsize('val.bin') / (1024 * 1024):.2f} MB")