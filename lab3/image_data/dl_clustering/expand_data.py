import pickle
import os
import numpy as np
from tqdm import tqdm
import torch

config = {
    'all': ['enrico_image_data.pkl', 'enrico_image_triplet.pkl'],
}

mode = "all"

pkl_file = os.path.join("./image_data/" + config[mode][0])
#read the pkl file
with open(pkl_file, 'rb') as f:
    corpus = pickle.load(f)
        
# split the corpus according to the "label_id" value in the corpus
# create a dictionary to store the split corpus
corpus_dict = {}
for element in corpus:
    label_id = element['label_id']
    if label_id not in corpus_dict:
        corpus_dict[label_id] = []
    corpus_dict[label_id].append(element)

def to_torch_tensor(embedding):
    # embedding to float 32
    embedding = embedding.astype(np.float32)
    # embedding to torch tensor
    embedding = torch.from_numpy(embedding)
    # set embedding require grad
    embedding.requires_grad = True
    # set embedding to float
    embedding = embedding.float()
    return embedding


expanded_corpus = []
for key, value in tqdm(corpus_dict.items()):
    # iteratively go through the val in value list, trace back when the total count is less than 1000
    count = 0
    indexer, total_count = 0, len(value)
    while True:
        if count < 1000:
            # get the element by indexer
            element = value[indexer]
            anchor_embedding = element['vector']
            anchor_embedding = to_torch_tensor(anchor_embedding)
            # get positive embedding, randaomly choose an index from the list not itself
            positive_index = np.random.choice([i for i in range(total_count) if i != indexer])
            positive_embedding = value[positive_index]['vector']
            positive_embedding = to_torch_tensor(positive_embedding)
            # get negative embedding, randomly choose a key from the corpus_dict
            # first, randomly choose a key not the current key
            negative_key = np.random.choice([k for k in corpus_dict.keys() if k != key])
            # second, randomly choose an index from the list
            negative_index = np.random.choice([i for i in range(len(corpus_dict[negative_key]))])
            negative_embedding = corpus_dict[negative_key][negative_index]['vector']
            negative_embedding = to_torch_tensor(negative_embedding)
            choice_str = str(key) + "_" + str(indexer) + "_" + str(positive_index) + "_" + str(negative_key) + "_" + str(negative_index)
            # append to the list
            expanded_corpus.append({"class": str(key) + "-" + str(count), "choice":choice_str, "anchor_embedding": anchor_embedding, "positive_embedding": positive_embedding, "negative_embedding": negative_embedding})
            if indexer == total_count - 1:
                indexer = 0
            else:
                indexer += 1
            count += 1
        else:
            break

# save expanded_corpus as pkl file
with open(os.path.join("./image_data/" + config[mode][1]), 'wb') as f:
    pickle.dump(expanded_corpus, f)

