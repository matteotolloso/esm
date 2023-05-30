import torch
import esm
import json
from Bio.Seq import Seq
import sys
import numpy as np
import multiprocessing
import os
import time

# ********* SETTINGS **********


FILE_PATH = "./dataset/emoglobina.json"



ANNOTATION_KEY = "esm"

# *****************************


def predict(id, query_sequence):

    # evaluate the time
    start = time.time()

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)

    data = [(id, query_sequence)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  # contains the lens of each sequence


    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    
    z = np.array(token_representations[0, 1 : batch_lens[0] - 1]).tolist() 

    end = time.time()

    print(f"Time for embedding: {end - start}", file=sys.stderr)


    seq_dict[id][ANNOTATION_KEY] = z

    # start = time.time()
    # with open(FILE_PATH, "r") as file:   # load the list of seqrecords alreay annotated with the others embeddings
    #     seq_dict = json.load(file)
    #     seq_dict[id][ANNOTATION_KEY] = z
    # with open(FILE_PATH, "w") as file:   # save the list of seqrecords alreay annotated with the others embeddings
    #     json.dump(seq_dict, file, indent=4)
    # end = time.time()
    # print(f"Time for saving: {end - start}", file=sys.stderr)

    print("element added", file=sys.stderr)
    return




pid = os.getpid()
print(f'{pid}, {FILE_PATH}', file=sys.stderr)


seq_dict = []
    
with open(FILE_PATH, "r") as file:   # load the list of seqrecords alreay annotated with the others embeddings
    seq_dict = json.load(file)


for id in seq_dict.keys():

    if ANNOTATION_KEY in seq_dict[id]:
        print(f"key: {id} already embedded", file=sys.stderr)
        continue

    seq_string = seq_dict[id]["sequence"]

    seq_string = seq_string.replace(" ", "").replace("\n", "")

    if set(seq_string).issubset(set(["A", "C", "G", "T"])):
        seq_string = str(Seq(seq_string).translate(stop_symbol=""))
        print("The nucleotides sequence for ", id, " has been translated", file=sys.stderr)

    print("Predicting the embedding for ", id, "...", file=sys.stderr)

    # the code run in a different process to avoid memory leaks

    predict(id, seq_string)


    # p = multiprocessing.Process(target=predict, args=(id, seq_string, ))
    # p.start()
    # p.join()

with open(FILE_PATH, "w") as file:   # save the list of seqrecords alreay annotated with the others embeddings
    json.dump(seq_dict, file, indent=4)



# Look at the unsupervised self-attention map contact predictions
def vis():
    import matplotlib.pyplot as plt
    for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
        plt.matshow(attention_contacts[: tokens_len, : tokens_len])
        plt.title(seq)
        plt.show()