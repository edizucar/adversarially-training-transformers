import numpy as np
import os
import tiktoken
from tqdm import tqdm

dataset= "tiny_stories"
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')


enc = tiktoken.get_encoding("gpt2")


"""
bad_chars = set()
for token in tqdm(range(50147)):
    s = enc.decode([token])

    for c in s:
        if ord(c) > 256:
            bad_chars.add((c,ord(c)))
print(bad_chars)
"""

split = True
if split:
    block_size=1000
    for i in range(int(train_data.shape[0]/block_size)):
        print(f'----  {i*block_size}-{i*block_size + block_size} ---------')
        for j in range(block_size):
            print(enc.decode([train_data[i*block_size + j]]),end='')
        print()
        input()

else:
    bad_chars = set()
    tokens = set()

    for i,token in tqdm(enumerate(train_data)):
        if token not in tokens:
            s = enc.decode([token])

            for c in s:
                if ord(c) > 256:
                    bad_chars.add((c,ord(c)))
                    tokens.add(token)

    print(bad_chars)

    # {('•', 8226), ('™', 8482), ('†', 8224), ('€', 8364), ('…', 8230), ('š', 353), ('�', 65533), ('˜', 732), ('–', 8211)}



 # 33000 - 34,000