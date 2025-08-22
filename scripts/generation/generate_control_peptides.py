#!/usr/bin/env python3
import argparse
import gzip
import random
import sys
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

# Import mutation functionality
from peptide_mutations import PeptideMutator

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def is_valid_peptide(peptide: str) -> bool:
    """Check if peptide contains only standard amino acids."""
    return all(aa in AMINO_ACIDS for aa in peptide.upper())

def generate_random_peptides(length: int, count: int) -> List[str]:
    return ["".join(random.choices(AMINO_ACIDS, k=length)) for _ in range(count)]

def parse_fasta_sequences(fasta_path: Path) -> List[str]:
    """Parse FASTA sequences from plain text or gzipped files."""
    sequences = []
    seq = []
    
    # Determine if file is gzipped based on extension
    if str(fasta_path).endswith('.gz'):
        open_func = lambda: gzip.open(fasta_path, 'rt')  # 'rt' for text mode
    else:
        open_func = lambda: open(fasta_path, 'r')
    
    with open_func() as f:
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
    print(f"Parsing FASTA file: {fasta_path}")
    sequences = parse_fasta_sequences(fasta_path)
    
    print(f"Extracting {length}-mer peptides from {len(sequences)} proteins...")
    all_subseqs = set()  # Use set to automatically collapse duplicates
    invalid_peptides = 0
    
    # Use progress bar for subsequence extraction
    for seq in tqdm(sequences, desc="Processing proteins", unit="protein"):
        if len(seq) >= length:
            for i in range(len(seq) - length + 1):
                peptide = seq[i:i+length]
                if is_valid_peptide(peptide):
                    all_subseqs.add(peptide)
                else:
                    invalid_peptides += 1
    
    if not all_subseqs:
        raise ValueError(f"No valid subsequences of length {length} found in {fasta_path}")
    
    if invalid_peptides > 0:
        print(f"Filtered out {invalid_peptides} peptides with non-standard amino acids")
    
    print(f"Found {len(all_subseqs)} valid unique {length}-mer peptides")
    
    # Convert set back to list for sampling
    unique_subseqs = list(all_subseqs)
    
    # Ensure sampling without replacement - if not enough unique peptides, inform user
    if count > len(unique_subseqs):
        print(f"Warning: Requested {count} peptides, but only {len(unique_subseqs)} valid unique peptides available. Returning all {len(unique_subseqs)} valid peptides.", file=sys.stderr)
        return unique_subseqs
    
    # Sample without replacement
    print(f"Sampling {count} peptides without replacement...")
    return random.sample(unique_subseqs, k=count)

def generate_llm_peptides(length: int, count: int, model_name: str = "protgpt2", top_k: int = 950, top_p: float = 0.9, repetition_penalty: float = 1.2) -> List[str]:
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
            sequences = llm_pipeline(prompt, max_length=max_length, do_sample=True, top_k=top_k, top_p=top_p, temperature=1.0, repetition_penalty=repetition_penalty, num_return_sequences=min(count - len(peptides), batch_size), eos_token_id=0)
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
                if len(pep) == length and is_valid_peptide(pep):
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
                        
                        # Apply fixed temperature
                        predictions = predictions / 1.0
                        
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

                        if len(final_peptide) == length and is_valid_peptide(final_peptide):
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

def generate_fake_proteome_lengths(num_proteins: int, reference_fasta_path: Path) -> List[int]:
    """Generate protein lengths matching the distribution from a reference proteome."""
    # Parse reference proteome to get length distribution
    sequences = parse_fasta_sequences(reference_fasta_path)
    reference_lengths = [len(seq) for seq in sequences]
    
    # Sample from the empirical distribution
    sampled_lengths = np.random.choice(reference_lengths, size=num_proteins, replace=True)
    return sampled_lengths.tolist()

def get_user_input(prompt: str) -> str:
    """Get user input with a prompt."""
    return input(prompt).strip()

def get_existing_proteome_path() -> Path:
    """Interactive prompt to get existing proteome file path."""
    while True:
        path_str = get_user_input("Please provide the path to your ProtGPT2-generated proteome file: ")
        # Strip quotes that users might add
        path_str = path_str.strip('"\'')
        path = Path(path_str)
        if path.exists() and path.is_file():
            return path
        else:
            print(f"Error: File '{path}' not found. Please try again.")

def configure_proteome_generation():
    """Interactive configuration for new proteome generation using ProtGPT2 only."""
    print("\nConfiguring new ProtGPT2 proteome generation...")
    print("This will generate completely synthetic proteins using ProtGPT2.")
    
    # Get number of proteins
    while True:
        try:
            num_proteins = int(get_user_input("How many proteins should be generated? "))
            if num_proteins > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid integer.")
    
    # Get protein length range
    print("\nSpecify the protein length range:")
    while True:
        try:
            min_len = int(get_user_input("Minimum protein length: "))
            max_len = int(get_user_input("Maximum protein length: "))
            if min_len > 0 and max_len >= min_len:
                break
            else:
                print("Please ensure minimum > 0 and maximum >= minimum.")
        except ValueError:
            print("Please enter valid integers.")
    
    # Generate uniform random lengths in the specified range
    target_lengths = [random.randint(min_len, max_len) for _ in range(num_proteins)]
    
    print(f"\nWill generate {num_proteins} proteins with lengths between {min_len}-{max_len} amino acids.")
    return num_proteins, target_lengths

def generate_fake_proteome(num_proteins: int, target_lengths: List[int], model_name: str = "protgpt2") -> List[str]:
    """Generate a fake proteome with specified protein lengths using LLM."""
    try:
        from transformers import pipeline
    except ImportError:
        print("Error: transformers package is required for fake proteome generation. Please install with 'pip install transformers torch'", file=sys.stderr)
        sys.exit(1)
    
    print(f"Generating {num_proteins} proteins using {model_name}...")
    
    if model_name.lower() == "protgpt2":
        model_id = "nferruz/ProtGPT2"
        llm_pipeline = pipeline('text-generation', model=model_id, framework="pt")
        
        proteins = []
        # Use progress bar for protein generation
        for i, target_length in enumerate(tqdm(target_lengths, desc="Generating proteins", unit="protein")):
            
            # Generate protein of approximately target length
            max_tokens = max(10, target_length // 4)  # ProtGPT2 tokens are ~4 amino acids
            tries = 0
            protein = None
            
            while tries < 5:  # Try up to 5 times to get reasonable length
                try:
                    # Start with empty prompt to get natural protein start
                    sequences = llm_pipeline(
                        "", 
                        max_length=max_tokens, 
                        do_sample=True, 
                        top_k=950, 
                        top_p=0.9, 
                        temperature=1.0, 
                        repetition_penalty=1.2, 
                        num_return_sequences=1,
                        eos_token_id=0
                    )
                    
                    if sequences and len(sequences) > 0:
                        gen_text = sequences[0].get('generated_text', '')
                        # Clean the sequence - keep only valid amino acids
                        clean_seq = ''.join([c for c in gen_text.upper() if c in AMINO_ACIDS])
                        
                        if len(clean_seq) >= 50:  # Minimum reasonable protein length
                            # Trim or extend to approximate target length
                            if len(clean_seq) > target_length * 1.5:
                                clean_seq = clean_seq[:target_length]
                            protein = clean_seq
                            break
                
                except Exception as e:
                    print(f"Warning: Error generating protein {i+1}: {e}", file=sys.stderr)
                
                tries += 1
            
            # Fallback to random sequence if generation failed
            if protein is None:
                protein = ''.join(random.choices(AMINO_ACIDS, k=target_length))
            
            proteins.append(protein)
        
        return proteins
    
    else:
        print(f"Error: Unsupported model '{model_name}' for proteome generation. Only protgpt2 is supported.", file=sys.stderr)
        sys.exit(1)

def write_fasta(peptides: List[str], output_path: Path, prefix: str = "peptide"):
    print(f"Writing {len(peptides)} peptides to {output_path}...")
    with open(output_path, 'w') as f:
        for i, pep in enumerate(tqdm(peptides, desc="Writing peptides", unit="peptide")):
            f.write(f">{prefix}_{i+1}\n{pep}\n")
    print(f"✅ Successfully wrote {len(peptides)} peptides to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate control peptides for neoantigen analysis.")
    parser.add_argument('--length', type=int, required=True, help='Peptide length (e.g., 8, 9, 10)')
    parser.add_argument('--count', type=int, required=True, help='Number of peptides to generate')
    parser.add_argument('--source', choices=['random', 'fasta', 'llm'], required=True, help='Source of peptides: random, fasta, or llm')
    parser.add_argument('--llm_model', choices=['protgpt2', 'esm2'], default='protgpt2', help='LLM model to use for generation (protgpt2 or esm2)')
    parser.add_argument('--fasta_file', type=Path, help='Path to reference FASTA file (required if source is fasta)')
    parser.add_argument('--output', type=Path, default=Path('control_peptides.fasta'), help='Output FASTA file')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility (not used for llm models)')
    
    # Mutation arguments
    parser.add_argument('--mutate', action='store_true', help='Apply mutations to generated peptides')
    parser.add_argument('--mutations', type=int, default=1, help='Number of mutations to apply per peptide (default: 1)')

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
        # Choose workflow based on the LLM model
        if args.llm_model.lower() == 'protgpt2':
            # Interactive proteome workflow for ProtGPT2 (to avoid M bias)
            print(f"\nGenerating {args.count} peptides of length {args.length} using ProtGPT2 proteome approach...")
            print("This approach generates a fake proteome first, then samples peptides from it.")
        elif args.llm_model.lower() == 'esm2':
            # Direct generation for ESM2 (no M bias issue)
            print(f"\nGenerating {args.count} peptides of length {args.length} using ESM2 direct generation...")
            peptides = generate_llm_peptides(args.length, args.count, args.llm_model, args.top_k, args.top_p, args.repetition_penalty)
        else:
            print(f"Error: Unsupported LLM model '{args.llm_model}'", file=sys.stderr)
            sys.exit(1)
        
        # Only run interactive proteome workflow for ProtGPT2
        if args.llm_model.lower() == 'protgpt2':
            # Check if user has existing proteome
            has_existing = get_user_input("\nDo you have an existing ProtGPT2-generated proteome file? (y/n): ").lower().startswith('y')
            
            if has_existing:
                proteome_path = get_existing_proteome_path()
                print(f"Using existing proteome: {proteome_path}")
            else:
                # Ask if user wants to generate new proteome
                generate_new = get_user_input("Would you like to generate a new fake proteome? (y/n): ").lower().startswith('y')
                
                if not generate_new:
                    print("Cannot proceed without a proteome. Exiting.")
                    sys.exit(1)
                
                # Configure proteome generation (no reference needed)
                num_proteins, target_lengths = configure_proteome_generation()
                
                # Generate the fake proteome
                fake_proteins = generate_fake_proteome(num_proteins, target_lengths, args.llm_model)
                
                # Save the generated proteome
                proteome_output = Path(f'fake_proteome_{num_proteins}proteins.fasta')
                write_fasta(fake_proteins, proteome_output, prefix="protein")
                print(f"\nGenerated fake proteome saved to: {proteome_output}")
                proteome_path = proteome_output
            
            # Now sample peptides from the proteome using FASTA method
            print(f"\nSampling {args.count} peptides from the proteome...")
            peptides = sample_peptides_from_fasta(proteome_path, args.length, args.count)
    else:
        print(f"Unknown source: {args.source}", file=sys.stderr)
        sys.exit(1)

    # Save initial peptides
    write_fasta(peptides, args.output)
    print(f"Wrote {len(peptides)} peptides to {args.output}")
    
    # Handle mutations (either from command line flag or interactive prompt)
    should_mutate = args.mutate
    mutations_count = args.mutations
    
    # If not specified via command line, ask interactively
    if not args.mutate:
        try:
            response = input("\n🧬 Would you like to mutate these peptides? (y/n): ").strip().lower()
            if response.startswith('y'):
                should_mutate = True
                while True:
                    try:
                        mutations_count = int(input("How many mutations per peptide? (default: 1): ").strip() or "1")
                        if mutations_count > 0:
                            break
                        else:
                            print("Please enter a positive number.")
                    except ValueError:
                        print("Please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            print("\nNo mutations applied.")
            should_mutate = False
    
    # Apply mutations if requested
    if should_mutate:
        print(f"\n🧬 Applying {mutations_count} mutation(s) per peptide...")
        
        # Initialize the mutator (no arguments needed - uses hardcoded data)
        mutator = PeptideMutator()
        
        # Apply mutations to all peptides
        mutated_peptides = mutator.mutate_peptide_list(peptides, mutations_count)
        
        # Update output filename to indicate mutations
        output_path = args.output
        if not str(output_path).endswith('_mutated.fasta'):
            # Insert 'mutated' before the file extension
            stem = output_path.stem
            suffix = output_path.suffix
            output_path = output_path.with_name(f"{stem}_mutated_{mutations_count}x{suffix}")
        
        write_fasta(mutated_peptides, output_path, prefix=f"{args.source}_mutated")
        print(f"✅ Mutated peptides saved to: {output_path}")

if __name__ == "__main__":
    main() 