import torch
import esm
import json
from Bio.Seq import Seq
import sys
import numpy as np
import os
import time
from Bio import SeqIO


# ********* SETTINGS **********

FASTA_FILE_PATH = "./dataset/batterio.fasta"
OUT_DIR = './dataset/batterio_esm'
MAX_CHUNK_SIZE = 1024

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
    print(f'{pid}, {FASTA_FILE_PATH}', file=sys.stderr)

    # check if the output directory exists
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)        

    for count, seqrecord in enumerate(SeqIO.parse(FASTA_FILE_PATH, "fasta")):

        seq_id = seqrecord.id

        # check if the file already exists
        if os.path.exists(os.path.join(OUT_DIR, f"{seq_id}.npy")):
            print(f"Skipping {seq_id} because already exists", file=sys.stderr)
            continue

        seq_string = str(seqrecord.seq)
        seq_string = seq_string.replace(" ", "").replace("\n", "")

        if set(seq_string).issubset(set(["A", "C", "G", "T"])):
            seq_string = str(Seq(seq_string).translate(stop_symbol=""))
            print("The nucleotides sequence for ", seq_id, " has been translated", file=sys.stderr)
        
        # split the sequence in chunks such that each chunk has approximately the same length
        N = int(np.ceil(len(seq_string) / MAX_CHUNK_SIZE)) # number of chunks
        chunks = [seq_string[(i*len(seq_string))//N:((i+1)*len(seq_string))//N] for i in range(N)] # list of chunks
        
        sequence_embedding = []
        for chunk_index, chunk in enumerate(chunks):
            print(f"Predicting the embedding {count+1}, chunk {chunk_index+1}/{len(chunks)}", file=sys.stderr)
            z = predict(seq_id, chunk)
            sequence_embedding.append(z)
        
        sequence_embedding = np.array(sequence_embedding)

        # save the embedding
        np.save(os.path.join(OUT_DIR, f"{seq_id}.npy"), sequence_embedding)

    print(f'{pid}, DONE', file=sys.stderr)


if __name__ == "__main__":
    main()
