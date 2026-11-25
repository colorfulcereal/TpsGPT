import torch
import esm
from transformers import AutoTokenizer, EsmForProteinFolding
import biotite.structure.io as bsio
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True

def get_esm_model():
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model.esm = model.esm.half()
    model.trunk.set_chunk_size(8)

    device = None
    # Check if MPS is available (Metal Performance Shaders)
    if torch.backends.mps.is_available():
        print("MPS is available!")
        device = torch.device('mps') # Apple M1/M2 machines
    elif torch.cuda.is_available():
        print("CUDA is available!")
        device = torch.device('cuda')
    else:
        print("Running on CPU")

    if device is not None:
        model = model.to(device)

    model = model.eval()
    return model

def pLDDT(sequence, model):
    with torch.no_grad():
        output = model.infer_pdb(sequence)
    with open("result.pdb", "w") as f:
        f.write(output)

    struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
    return struct.b_factor.mean()  # this will be the pLDDT

def parse_fasta(fasta_path):
    """
    Parses a FASTA file and yields (seq_id, sequence) tuples.
    """
    with open(fasta_path, "r") as f:
        seq_id = None
        seq_chunks = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id:
                    yield seq_id, "".join(seq_chunks)
                seq_id = line[1:].split()[0]  # take first word after '>'
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if seq_id:
            yield seq_id, "".join(seq_chunks)

input_fasta_file = '/teamspace/studios/this_studio/training_files/all_filtered_tps_seq.fasta'
out_file = Path('/teamspace/studios/this_studio/training_files/pLDDT_scores_2800.txt')

model = get_esm_model()

if out_file.exists():
    print(f"Skipping {seq_id}: {out_file} already exists")
else:
    with open(out_file, "w") as f:
        for seq_id, sequence in parse_fasta(input_fasta_file):
            pLDDT_score = pLDDT(sequence, model)
            f.write(str(pLDDT_score))
            f.write('\n')
            f.flush()
            print(f"finished pLDDT {seq_id}")