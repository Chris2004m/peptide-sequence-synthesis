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

def generate_llm_peptides(length: int, count: int, model_name: str = "protgpt2", temperature: float = 1.0, top_k: int = 950, top_p: float = 0.9, repetition_penalty: float = 1.2) -> List[str]:
    try:
        from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForMaskedLM
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("Error: transformers and torch packages are required for LLM generation. Please install with 'pip install transformers torch'", file=sys.stderr)
        sys.exit(1)
    # Model configurations
    if model_name.lower() == "protgpt2":
        model_id = "nferruz/ProtGPT2"
        # Each token is ~4 amino acids, so for a peptide of length N, set max_length ≈ N/4 (rounded up)
        max_length = max(5, (length + 3) // 4)  # ensure at least 1 token
        prompt = "<|endoftext|>"
        use_pipeline = True
    elif model_name.lower() == "esm2":
        model_id = "facebook/esm2_t12_35M_UR50D"  # ESM-2 35M model (faster, smaller)
        max_length = length + 10  # ESM works with direct amino acid sequences
        use_pipeline = False
    else:
        print(f"Error: Unsupported model '{model_name}'. Supported models: protgpt2, esm2", file=sys.stderr)
        sys.exit(1)
    
    peptides = []
    tries = 0
    batch_size = min(50, count) if model_name.lower() == "protgpt2" else min(10, count)
    
    if use_pipeline:
        # ProtGPT2 pipeline approach
        llm_pipeline = pipeline('text-generation', model=model_id, framework="pt")
        while len(peptides) < count and tries < count * 10:
            sequences = llm_pipeline(prompt, max_length=max_length, do_sample=True, top_k=top_k, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty, num_return_sequences=min(count - len(peptides), batch_size), eos_token_id=0)
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
                    # Real-time progress output every 100 accepted peptides or at the end
                    if len(peptides) % 100 == 0 or len(peptides) == count:
                        print(f"[LLM] Generated {len(peptides)}/{count} valid peptides...", flush=True)
            tries += 1
    else:
        # ESM-2 approach using masked language modeling
        print(f"Loading ESM-2 model (this may take a moment)...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        
        while len(peptides) < count and tries < count * 10:  # Reduce tries for efficiency
            try:
                for _ in range(min(count - len(peptides), batch_size)):
                    # Simple approach: create a sequence with random masks
                    # Start with a random seed and mask 40% of positions
                    seed_length = max(1, length // 3)
                    seed_seq = "".join(random.choices(AMINO_ACIDS, k=seed_length))
                    
                    # Create masked sequence: seed + masks for remaining positions
                    remaining_masks = length - seed_length
                    masked_sequence = seed_seq + "<mask>" * remaining_masks
                    
                    # Tokenize
                    inputs = tokenizer(masked_sequence, return_tensors="pt", max_length=512, truncation=True)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        predictions = outputs.logits
                        
                        # Apply temperature
                        predictions = predictions / temperature
                        
                        # Find mask positions and generate amino acids
                        mask_token_id = tokenizer.mask_token_id
                        input_ids = inputs.input_ids[0]
                        
                        generated_sequence = []
                        token_idx = 0
                        
                        for token_id in input_ids:
                            if token_id == mask_token_id:
                                # Get prediction for this mask
                                logits = predictions[0, token_idx]
                                
                                # Simple top-k sampling
                                if top_k > 0 and top_k < logits.size(-1):
                                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                                    # Sample from top-k
                                    probs = F.softmax(top_k_logits, dim=-1)
                                    sampled_idx = torch.multinomial(probs, 1).item()
                                    sampled_token_id = top_k_indices[sampled_idx].item()
                                else:
                                    # Sample from full distribution
                                    probs = F.softmax(logits, dim=-1)
                                    sampled_token_id = torch.multinomial(probs, 1).item()
                                
                                sampled_token = tokenizer.decode([sampled_token_id])
                                
                                # Only add valid amino acids
                                if sampled_token in AMINO_ACIDS:
                                    generated_sequence.append(sampled_token)
                                else:
                                    # Fallback to random amino acid
                                    generated_sequence.append(random.choice(AMINO_ACIDS))
                            else:
                                # Keep original token if it's an amino acid
                                original_token = tokenizer.decode([token_id])
                                if original_token in AMINO_ACIDS:
                                    generated_sequence.append(original_token)
                            
                            token_idx += 1
                        
                        # Create final peptide
                        final_peptide = "".join(generated_sequence)
                        
                        # Ensure exact length
                        # Ensure exact length and record peptide
                        if len(final_peptide) >= length:
                            final_peptide = final_peptide[:length]
                        else:
                            final_peptide += "".join(random.choices(AMINO_ACIDS, k=length - len(final_peptide)))

                        if len(final_peptide) == length:
                            peptides.append(final_peptide)
                            if len(peptides) % 100 == 0 or len(peptides) == count:
                                print(f"[LLM] Generated {len(peptides)}/{count} valid peptides...", flush=True)
            
            except Exception as e:
                print(f"[LLM] Warning: ESM-2 generation error: {e}", file=sys.stderr)
                fallback_peptide = "".join(random.choices(AMINO_ACIDS, k=length))
                peptides.append(fallback_peptide)
                if len(peptides) % 100 == 0 or len(peptides) == count:
                    print(f"[LLM] Generated {len(peptides)}/{count} valid peptides...", flush=True)
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
    parser.add_argument('--source', choices=['random', 'fasta', 'llm'], required=True, help='Source of peptides: random, fasta, or llm')
    parser.add_argument('--llm_model', choices=['protgpt2', 'esm2'], default='protgpt2', help='LLM model to use for generation (protgpt2 or esm2)')
    parser.add_argument('--fasta_file', type=Path, help='Path to reference FASTA file (required if source is fasta)')
    parser.add_argument('--output', type=Path, default=Path('control_peptides.fasta'), help='Output FASTA file')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility (not used for llm models)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for LLM generation (higher = more random)')
    parser.add_argument('--top_k', type=int, default=950, help='Top-k sampling for LLM')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling for LLM')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition penalty for LLM')
    args = parser.parse_args()

    # Input validation
    if args.length < 1 or args.length > 50:
        print('Error: Peptide length must be between 1 and 50 amino acids', file=sys.stderr)
        sys.exit(1)
    if args.count < 1 or args.count > 10000000:
        print('Error: Count must be between 1 and 10,000,000 peptides', file=sys.stderr)
        sys.exit(1)
    if args.source == 'llm':
        if args.temperature <= 0:
            print('Error: Temperature must be positive', file=sys.stderr)
            sys.exit(1)
        if args.top_k < 1:
            print('Error: Top-k must be at least 1', file=sys.stderr)
            sys.exit(1)
        if args.top_p <= 0 or args.top_p > 1:
            print('Error: Top-p must be between 0 and 1', file=sys.stderr)
            sys.exit(1)
        if args.repetition_penalty <= 0:
            print('Error: Repetition penalty must be positive', file=sys.stderr)
            sys.exit(1)

    if args.seed is not None and args.source != 'llm':
        random.seed(args.seed)

    if args.source == 'random':
        peptides = generate_random_peptides(args.length, args.count)
    elif args.source == 'fasta':
        if not args.fasta_file:
            print('Error: --fasta_file is required when source is fasta', file=sys.stderr)
            sys.exit(1)
        peptides = sample_peptides_from_fasta(args.fasta_file, args.length, args.count)
    elif args.source == 'llm':
        peptides = generate_llm_peptides(args.length, args.count, args.llm_model, args.temperature, args.top_k, args.top_p, args.repetition_penalty)
    else:
        print(f"Unknown source: {args.source}", file=sys.stderr)
        sys.exit(1)

    write_fasta(peptides, args.output)
    print(f"Wrote {len(peptides)} peptides to {args.output}")

if __name__ == "__main__":
    main() 