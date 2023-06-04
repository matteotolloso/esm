import torch
import esm
import json
from Bio.Seq import Seq
import sys
import numpy as np
import os
import time

# ********* SETTINGS **********


FILE_PATH = "./dataset/topo.json"
MAX_CHUNK_SIZE = 1024
SAVE_EVERY = 100
ANNOTATION_KEY = "esmfold_650M"

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

    return z



def main():

    pid = os.getpid()
    print(f'{pid}, {FILE_PATH}', file=sys.stderr)
        
    file = open(FILE_PATH, "r")    # load the list of seqrecords alreay annotated with the others embeddings
    seq_dict = json.load(file)
    file.close()

    for seq_index, id in enumerate(seq_dict.keys()):

        if ANNOTATION_KEY in seq_dict[id]:
            print(f"key: {id} already embedded", file=sys.stderr)
            continue

        seq_string = seq_dict[id]["sequence"]

        seq_string = seq_string.replace(" ", "").replace("\n", "")

        if set(seq_string).issubset(set(["A", "C", "G", "T"])):
            seq_string = str(Seq(seq_string).translate(stop_symbol=""))
            print("The nucleotides sequence for ", id, " has been translated", file=sys.stderr)
        
        # split the sequence in chunks such that each chunk has approximately the same length
        N = int(np.ceil(len(seq_string) / MAX_CHUNK_SIZE)) # number of chunks
        chunks = [seq_string[(i*len(seq_string))//N:((i+1)*len(seq_string))//N] for i in range(N)] # list of chunks
        
        sequence_embedding = []
        for chunk_index, chunk in enumerate(chunks):
            print(f"Predicting the embedding for {id} ({seq_index+1}\{len(seq_dict.keys())}),\
                   chunk {chunk_index+1}/{len(chunks)}", file=sys.stderr)
            z = predict(id, chunk)
            sequence_embedding.append(z)
        
        seq_dict[id][ANNOTATION_KEY] = sequence_embedding

        # save the embedding every 100 sequences or at the end
        if (seq_index + 1) % SAVE_EVERY == 0 or seq_index == len(seq_dict.keys()) - 1:
            print(f"Dumping the results", file=sys.stderr)
            start = time.time()
            with open(FILE_PATH, "w") as file:	
                json.dump(seq_dict, file, indent=4)
            end = time.time()
            print(f"Dump executed in {end - start}s", file=sys.stderr)



if __name__ == "__main__":
    main()
