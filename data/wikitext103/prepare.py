import os
import requests
import tiktoken
import numpy as np
import zipfile
import io
import pickle

# specify the url and filename
url = 'https://wikitext.smerity.com/wikitext-103-raw-v1.zip'
filename = 'wikitext-103-v1.zip'

# download the wikitext-103 dataset
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(path='data/wikitext103/')

# specify the data file paths (train, test, valid)
train_data_path = 'data/wikitext103/wikitext-103-raw/wiki.train.raw'
test_data_path = 'data/wikitext103/wikitext-103-raw/wiki.test.raw'
valid_data_path = 'data/wikitext103/wikitext-103-raw/wiki.valid.raw'

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
#enc = tiktoken.encoding_for_model("gpt-4")

# define a function to load, tokenize and save data
def process_data(file_path, save_file):
    with open(file_path, 'r') as f:
        data = f.read()

    ids = enc.encode_ordinary(data)
    print(f"{file_path} has {len(ids):,} tokens")

    #export to bin files
    #ids = np.array(ids, dtype=np.uint16)
    ids = np.array(ids).astype(np.uint16)
    ids.tofile(save_file)

print(enc.n_vocab)

# process and save the tokenized train, test, valid data
process_data(train_data_path, 'data/wikitext103/train.bin')
process_data(test_data_path, 'data/wikitext103/test.bin')
process_data(valid_data_path, 'data/wikitext103/val.bin')