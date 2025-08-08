#!/usr/bin/env python3
"""
Generate 1M peptide datasets for algorithm benchmarking.
Creates 8 datasets: Random method (8-11mers) and Reference proteome method (8-11mers).
"""

import sys
import time
from pathlib import Path
from typing import List

# Add the scripts directory to the path so we can import from generate_control_peptides
sys.path.append(str(Path(__file__).parent))
from generate_control_peptides import generate_random_peptides, sample_peptides_from_fasta, write_fasta

def generate_all_datasets():
    """Generate all 8 required peptide datasets."""
    
    # Configuration
    base_dir = Path("/Users/chris/Desktop/Griffith Lab/Peptide Sequence Synthesis")
    output_dir = base_dir / "data" / "1M_Peptides"
    reference_fasta = base_dir / "data" / "protein.faa"
    
    lengths = [8, 9, 10, 11]
    count = 1_000_000  # 1 million peptides per dataset
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🧬 Generating 1M peptide datasets")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Reference proteome: {reference_fasta}")
    print(f"🔢 Lengths: {lengths}")
    print(f"📈 Count per dataset: {count:,}")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # Generate random method datasets
    print("\n🎲 RANDOM METHOD DATASETS")
    print("-" * 40)
    
    for length in lengths:
        print(f"\n🔄 Generating random {length}-mer peptides...")
        start_time = time.time()
        
        peptides = generate_random_peptides(length, count)
        
        # Remove duplicates and ensure we have exactly 1M unique peptides
        unique_peptides = list(set(peptides))
        while len(unique_peptides) < count:
            # Generate more peptides to replace duplicates
            additional_needed = count - len(unique_peptides)
            additional_peptides = generate_random_peptides(length, additional_needed * 2)  # Generate extra to account for duplicates
            unique_peptides.extend(additional_peptides)
            unique_peptides = list(set(unique_peptides))
        
        # Take exactly 1M peptides
        final_peptides = unique_peptides[:count]
        
        # Save to file
        output_file = output_dir / f"random_{length}mer_1M.fasta"
        write_fasta(final_peptides, output_file, prefix=f"random_{length}mer")
        
        elapsed = time.time() - start_time
        print(f"✅ Saved {len(final_peptides):,} unique {length}-mer peptides to {output_file.name}")
        print(f"⏱️  Time: {elapsed:.1f} seconds")
    
    # Generate reference proteome method datasets
    print(f"\n🧪 REFERENCE PROTEOME METHOD DATASETS")
    print("-" * 50)
    
    for length in lengths:
        print(f"\n🔄 Sampling {length}-mer peptides from reference proteome...")
        start_time = time.time()
        
        peptides = sample_peptides_from_fasta(reference_fasta, length, count)
        
        # Save to file
        output_file = output_dir / f"reference_{length}mer_1M.fasta"
        write_fasta(peptides, output_file, prefix=f"reference_{length}mer")
        
        elapsed = time.time() - start_time
        print(f"✅ Saved {len(peptides):,} unique {length}-mer peptides to {output_file.name}")
        print(f"⏱️  Time: {elapsed:.1f} seconds")
    
    total_elapsed = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print(f"🎉 ALL DATASETS GENERATED SUCCESSFULLY!")
    print(f"⏱️  Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"📁 Output directory: {output_dir}")
    print("\n📋 Generated files:")
    
    # List all generated files
    for length in lengths:
        random_file = output_dir / f"random_{length}mer_1M.fasta"
        reference_file = output_dir / f"reference_{length}mer_1M.fasta"
        print(f"   • {random_file.name}")
        print(f"   • {reference_file.name}")

if __name__ == "__main__":
    generate_all_datasets()
