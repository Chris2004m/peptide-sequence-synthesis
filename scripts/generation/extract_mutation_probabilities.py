#!/usr/bin/env python3
"""
Extract mutation probabilities from Cancer Hotspots data and generate hardcoded Python dictionary.
This is a utility script to convert the TSV data into hardcoded probabilities for the mutator.
"""

from pathlib import Path
import json

def extract_probabilities():
    """Extract and convert mutation data to probabilities."""
    
    mutation_file = Path("/Users/chris/Desktop/Griffith Lab/Peptide Sequence Synthesis/Cancer Hotspots SNP AA Changes.tsv")
    
    # Dictionary to store raw counts for each source amino acid
    raw_counts = {}
    
    with open(mutation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                continue
                
            substitution, count = parts[0], int(parts[1])
            if '>' not in substitution:
                continue
                
            source_aa, target_aa = substitution.split('>')
            
            # Initialize source AA dictionary if not exists
            if source_aa not in raw_counts:
                raw_counts[source_aa] = {}
            
            # Store the count for this substitution
            raw_counts[source_aa][target_aa] = count
    
    # Convert raw counts to normalized probabilities
    substitution_probabilities = {}
    
    for source_aa, targets in raw_counts.items():
        total_count = sum(targets.values())
        
        # Create lists for weighted random selection
        target_aas = list(targets.keys())
        probabilities = [count / total_count for count in targets.values()]
        
        substitution_probabilities[source_aa] = {
            'targets': target_aas,
            'probabilities': probabilities
        }
    
    return substitution_probabilities

def generate_hardcoded_dict():
    """Generate Python code for hardcoded probability dictionary."""
    
    probs = extract_probabilities()
    
    print("# Hardcoded mutation probabilities extracted from Cancer Hotspots data")
    print("# Generated from 1,615,206 real tumor mutations")
    print("MUTATION_PROBABILITIES = {")
    
    for source_aa in sorted(probs.keys()):
        data = probs[source_aa]
        print(f"    '{source_aa}': {{")
        print(f"        'targets': {data['targets']},")
        print(f"        'probabilities': {data['probabilities']}")
        print(f"    }},")
    
    print("}")
    
    # Also print some statistics
    print(f"\n# Statistics:")
    print(f"# - {len(probs)} source amino acids")
    total_substitutions = sum(len(data['targets']) for data in probs.values())
    print(f"# - {total_substitutions} unique substitution patterns")

if __name__ == "__main__":
    generate_hardcoded_dict()
