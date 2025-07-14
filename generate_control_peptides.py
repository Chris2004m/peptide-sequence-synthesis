#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path
from typing import List

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def generate_random_peptides(length: int, count: int) -> List[str]:
    return ["".join(random.choices(AMINO_ACIDS, k=length)) for _ in range(count)]

def parse_fasta_sequences(fasta_path: Path) -> List[str]:
    sequences = []
    seq = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq:
                    sequences.append(''.join(seq))
                    seq = []
            else:
                seq.append(line)
        if seq:
            sequences.append(''.join(seq))
    return sequences

def sample_peptides_from_fasta(fasta_path: Path, length: int, count: int) -> List[str]:
    sequences = parse_fasta_sequences(fasta_path)
    all_subseqs = []
    for seq in sequences:
        if len(seq) >= length:
            for i in range(len(seq) - length + 1):
                all_subseqs.append(seq[i:i+length])
    if not all_subseqs:
        raise ValueError(f"No subsequences of length {length} found in {fasta_path}")
    peptides = random.sample(all_subseqs, k=min(count, len(all_subseqs)))
    while len(peptides) < count:
        peptides.append(random.choice(all_subseqs))
    return peptides[:count]

def generate_protgpt2_peptides(length: int, count: int) -> List[str]:
    try:
        from transformers import pipeline
    except ImportError:
        print("Error: transformers package is required for ProtGPT2 generation. Please install with 'pip install transformers torch'", file=sys.stderr)
        sys.exit(1)
    # Each token is ~4 amino acids, so for a peptide of length N, set max_length ≈ N/4 (rounded up)
    max_length = max(5, (length + 3) // 4)  # ensure at least 1 token
    protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2", framework="pt")
    peptides = []
    tries = 0
    while len(peptides) < count and tries < count * 10:
        sequences = protgpt2("<|endoftext|>", max_length=max_length, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=min(count - len(peptides), 10), eos_token_id=0)
        if not sequences or not hasattr(sequences, '__iter__'):
            tries += 1
            continue
        for seq in sequences:
            if not isinstance(seq, dict):
                continue
            gen_text = seq.get('generated_text', '')
            if not isinstance(gen_text, str):
                continue
            # Remove whitespace and newlines, keep only valid amino acids
            pep = ''.join([c for c in gen_text if c in AMINO_ACIDS])
            if len(pep) == length:
                peptides.append(pep)
        tries += 1
    if len(peptides) < count:
        print(f"Warning: Only generated {len(peptides)} peptides of requested {count} with exact length {length}.", file=sys.stderr)
    return peptides[:count]

def write_fasta(peptides: List[str], output_path: Path, prefix: str = "peptide"):
    with open(output_path, 'w') as f:
        for i, pep in enumerate(peptides, 1):
            f.write(f">{prefix}_{i}\n{pep}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate control peptides for neoantigen analysis.")
    parser.add_argument('--length', type=int, required=True, help='Peptide length (e.g., 8, 9, 10)')
    parser.add_argument('--count', type=int, required=True, help='Number of peptides to generate')
    parser.add_argument('--source', choices=['random', 'fasta', 'protgpt2'], required=True, help='Source of peptides: random, fasta, or protgpt2')
    parser.add_argument('--fasta_file', type=Path, help='Path to reference FASTA file (required if source is fasta)')
    parser.add_argument('--output', type=Path, default=Path('control_peptides.fasta'), help='Output FASTA file')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility (not used for protgpt2)')
    args = parser.parse_args()

    if args.seed is not None and args.source != 'protgpt2':
        random.seed(args.seed)

    if args.source == 'random':
        peptides = generate_random_peptides(args.length, args.count)
    elif args.source == 'fasta':
        if not args.fasta_file:
            print('Error: --fasta_file is required when source is fasta', file=sys.stderr)
            sys.exit(1)
        peptides = sample_peptides_from_fasta(args.fasta_file, args.length, args.count)
    elif args.source == 'protgpt2':
        peptides = generate_protgpt2_peptides(args.length, args.count)
    else:
        print(f"Unknown source: {args.source}", file=sys.stderr)
        sys.exit(1)

    write_fasta(peptides, args.output)
    print(f"Wrote {len(peptides)} peptides to {args.output}")

if __name__ == "__main__":
    main() 