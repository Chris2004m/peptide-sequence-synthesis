#!/usr/bin/env python3
"""
Peptide mutation module for applying biologically realistic amino acid substitutions.
Uses empirical substitution frequencies from cancer hotspots data.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Hardcoded mutation probabilities extracted from Cancer Hotspots data
# Generated from 1,615,206 real tumor mutations
MUTATION_PROBABILITIES = {
    'A': {
        'targets': ['D', 'E', 'G', 'P', 'S', 'T', 'V'],
        'probabilities': [0.06442224748268717, 0.03253762868728888, 0.03302940156883611, 0.033932328826758895, 0.12809474286727776, 0.3531654856055659, 0.3548181649615853]
    },
    'C': {
        'targets': ['F', 'G', 'R', 'S', 'W', 'Y'],
        'probabilities': [0.2530173040570016, 0.058649604963404586, 0.145072948475595, 0.1640734816538219, 0.06131549609810479, 0.3178711647520721]
    },
    'D': {
        'targets': ['A', 'E', 'G', 'H', 'N', 'V', 'Y'],
        'probabilities': [0.01911663216011042, 0.07991718426501035, 0.08929310854776694, 0.12206447796509909, 0.4668638469880706, 0.040629005225278514, 0.1821157448486641]
    },
    'E': {
        'targets': ['A', 'D', 'G', 'K', 'Q', 'V'],
        'probabilities': [0.02078597572722019, 0.15556411738264947, 0.05168560970911192, 0.5900404546330187, 0.15214794837218262, 0.02977589417581712]
    },
    'F': {
        'targets': ['C', 'I', 'L', 'S', 'V', 'Y'],
        'probabilities': [0.10893884227745237, 0.07113081103899531, 0.543163949131972, 0.09295024009287109, 0.1305472006754261, 0.0532689567832832]
    },
    'G': {
        'targets': ['A', 'C', 'D', 'E', 'R', 'S', 'V', 'W'],
        'probabilities': [0.05103914934751087, 0.08168940774064022, 0.11845187195598023, 0.22580956984050266, 0.21745176041937764, 0.12200617169201027, 0.13535338513588877, 0.04819868386808938]
    },
    'H': {
        'targets': ['D', 'L', 'N', 'P', 'Q', 'R', 'Y'],
        'probabilities': [0.05809586342744583, 0.06056467498358503, 0.13652002626395274, 0.047143795141168746, 0.12034143138542351, 0.15600787918581746, 0.4213263296126067]
    },
    'I': {
        'targets': ['F', 'K', 'L', 'M', 'N', 'R', 'S', 'T', 'V'],
        'probabilities': [0.08775578320367661, 0.016097974274050682, 0.07360907853860175, 0.2549231045726463, 0.07340368173765693, 0.009627975044288686, 0.0634419368918329, 0.1803640658296747, 0.24077639990757144]
    },
    'K': {
        'targets': ['E', 'I', 'M', 'N', 'Q', 'R', 'T'],
        'probabilities': [0.12881281544876014, 0.026434395097989567, 0.04554278286997181, 0.43652200334227986, 0.06578214412315794, 0.13698283283536739, 0.1599230262824733]
    },
    'L': {
        'targets': ['F', 'H', 'I', 'M', 'P', 'Q', 'R', 'S', 'V', 'W'],
        'probabilities': [0.2834173953263, 0.026749333744466225, 0.15621214457194457, 0.11165561747021122, 0.11706274915754465, 0.03705702266370064, 0.06639429112613704, 0.030967116710349536, 0.16400898619033985, 0.006475343039006233]
    },
    'M': {
        'targets': ['I', 'K', 'L', 'R', 'T', 'V'],
        'probabilities': [0.6131266983188509, 0.03723610737854221, 0.07895129744095077, 0.02275374003404103, 0.10310848338260324, 0.1448236734450118]
    },
    'N': {
        'targets': ['D', 'H', 'I', 'K', 'S', 'T', 'Y'],
        'probabilities': [0.12437431327066292, 0.0962641924062996, 0.09370040288121108, 0.22768892687095593, 0.2934012941032841, 0.09568428763276768, 0.0688865828348187]
    },
    'P': {
        'targets': ['A', 'H', 'L', 'Q', 'R', 'S', 'T'],
        'probabilities': [0.042324037455984155, 0.07960765550241451, 0.3351935746461359, 0.04310797736225456, 0.03377962457784864, 0.38095936206118035, 0.08508372839410235]
    },
    'Q': {
        'targets': ['E', 'H', 'K', 'L', 'P', 'R'],
        'probabilities': [0.08721896421845574, 0.19889734513274338, 0.09643893963205161, 0.048811881188118805, 0.02723977638619544, 0.07155567495767835]
    },
    'R': {
        'targets': ['C', 'G', 'H', 'I', 'K', 'L', 'M', 'P', 'Q', 'S', 'T', 'W'],
        'probabilities': [0.19088663832633346, 0.0341246551334329, 0.17900536518162424, 0.03915035983036549, 0.06095507855075889, 0.05716912043659549, 0.018371179939078156, 0.01772044728434504, 0.19506090012048194, 0.041279883227176916, 0.026271186440677964, 0.13176724481711702]
    },
    'S': {
        'targets': ['A', 'C', 'F', 'G', 'I', 'L', 'N', 'P', 'R', 'T', 'W', 'Y'],
        'probabilities': [0.017691176470588235, 0.0987373737373737, 0.30062267908902695, 0.0333954545454545, 0.05199734711493448, 0.22098484848484849, 0.07604040404040404, 0.048950757575757576, 0.0663510101010101, 0.03954924242424242, 0.006679292929292929, 0.09712563131313132]
    },
    'T': {
        'targets': ['A', 'I', 'K', 'M', 'N', 'P', 'R', 'S'],
        'probabilities': [0.20146739130434782, 0.2340043478260869, 0.06055434782608695, 0.24835652173913042, 0.08626086956521738, 0.06636959350180505, 0.029079710144927538, 0.10748913043478261]
    },
    'V': {
        'targets': ['A', 'D', 'E', 'F', 'G', 'I', 'L', 'M'],
        'probabilities': [0.16564842300556587, 0.023004319654427646, 0.03785388127853882, 0.07409270216962524, 0.06298969072164949, 0.24789389067524116, 0.17164092664092664, 0.226599861495845]
    },
    'W': {
        'targets': ['C', 'G', 'L', 'R', 'S'],
        'probabilities': [0.42157980456026056, 0.05529315960912052, 0.26587947882736157, 0.2007328990228013, 0.056514657980456025]
    },
    'Y': {
        'targets': ['C', 'D', 'F', 'H', 'N', 'S'],
        'probabilities': [0.4545847750865052, 0.05390967365967366, 0.11449567723342939, 0.27061011904761905, 0.0940843373493976, 0.04964797794117647]
    }
}


class PeptideMutator:
    """
    Class to handle biologically realistic peptide mutations based on empirical data.
    """
    
    def __init__(self):
        """
        Initialize the mutator with hardcoded cancer hotspots mutation data.
        No external files required.
        """
        self.substitution_probabilities = MUTATION_PROBABILITIES
        print(f"Initialized mutator with {len(self.substitution_probabilities)} amino acid substitution patterns")
        

        
    def _get_valid_substitution(self, source_aa: str) -> str:
        """
        Get a biologically realistic substitution for the source amino acid.
        
        Args:
            source_aa: The original amino acid to substitute
            
        Returns:
            A different amino acid based on empirical substitution frequencies
        """
        if source_aa not in self.substitution_probabilities:
            # If no data available for this AA, fall back to random substitution
            # ensuring it's different from source
            all_aas = "ACDEFGHIKLMNPQRSTVWY"
            available_aas = [aa for aa in all_aas if aa != source_aa]
            return random.choice(available_aas)
        
        # Use weighted random selection based on empirical frequencies
        data = self.substitution_probabilities[source_aa]
        
        # Normalize probabilities to ensure they sum to exactly 1.0 (fix floating point precision issues)
        probs = np.array(data['probabilities'])
        probs = probs / probs.sum()
        
        target_aa = np.random.choice(data['targets'], p=probs)
        
        # Ensure we don't return the same amino acid (shouldn't happen with real data)
        if target_aa == source_aa:
            all_aas = "ACDEFGHIKLMNPQRSTVWY"
            available_aas = [aa for aa in all_aas if aa != source_aa]
            return random.choice(available_aas)
            
        return target_aa
    
    def mutate_peptide(self, peptide: str, num_mutations: int = 1) -> str:
        """
        Apply mutations to a single peptide.
        
        Args:
            peptide: The peptide sequence to mutate
            num_mutations: Number of mutations to apply (default: 1)
            
        Returns:
            The mutated peptide sequence
        """
        if not peptide or len(peptide) == 0:
            return peptide
        
        # Convert to list for easy modification
        mutated_peptide = list(peptide)
        peptide_length = len(peptide)
        
        # Apply the specified number of mutations
        for _ in range(num_mutations):
            # Choose a random position
            position = random.randint(0, peptide_length - 1)
            original_aa = mutated_peptide[position]
            
            # Get a biologically realistic substitution
            new_aa = self._get_valid_substitution(original_aa)
            mutated_peptide[position] = new_aa
        
        return ''.join(mutated_peptide)
    
    def mutate_peptide_list(self, peptides: List[str], num_mutations: int = 1) -> List[str]:
        """
        Apply mutations to a list of peptides.
        
        Args:
            peptides: List of peptide sequences to mutate
            num_mutations: Number of mutations to apply per peptide (default: 1)
            
        Returns:
            List of mutated peptide sequences
        """
        print(f"Applying {num_mutations} mutation(s) per peptide to {len(peptides)} peptides...")
        
        mutated_peptides = []
        for peptide in peptides:
            mutated_peptide = self.mutate_peptide(peptide, num_mutations)
            mutated_peptides.append(mutated_peptide)
        
        return mutated_peptides


def load_peptides_from_fasta(fasta_path: Path) -> List[str]:
    """
    Load peptide sequences from a FASTA file.
    
    Args:
        fasta_path: Path to the FASTA file
        
    Returns:
        List of peptide sequences (without headers)
    """
    peptides = []
    
    with open(fasta_path, 'r') as f:
        current_sequence = []
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_sequence:
                    peptides.append(''.join(current_sequence))
                    current_sequence = []
            else:
                current_sequence.append(line)
        
        # Don't forget the last sequence
        if current_sequence:
            peptides.append(''.join(current_sequence))
    
    return peptides


def write_mutated_fasta(peptides: List[str], output_path: Path, prefix: str = "mutated_peptide"):
    """
    Write mutated peptides to a FASTA file.
    
    Args:
        peptides: List of peptide sequences
        output_path: Output file path
        prefix: Prefix for sequence headers
    """
    with open(output_path, 'w') as f:
        for i, peptide in enumerate(peptides, 1):
            f.write(f">{prefix}_{i:06d}\n")
            f.write(f"{peptide}\n")
    
    print(f"Wrote {len(peptides)} mutated peptides to {output_path}")


def mutate_existing_peptides_cli():
    """
    Command-line interface for mutating existing peptide FASTA files.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Apply biologically realistic mutations to existing peptide sequences"
    )
    parser.add_argument(
        "input_fasta", 
        type=Path,
        help="Input FASTA file containing peptides to mutate"
    )
    parser.add_argument(
        "output_fasta",
        type=Path, 
        help="Output FASTA file for mutated peptides"
    )
    parser.add_argument(
        "--mutations", 
        type=int,
        default=1,
        help="Number of mutations to apply per peptide (default: 1)"
    )

    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input_fasta.exists():
        print(f"Error: Input file {args.input_fasta} does not exist")
        return 1
    

    
    print(f"🧬 Mutating peptides from {args.input_fasta}")
    print(f"📊 Mutations per peptide: {args.mutations}")
    print(f"📁 Output file: {args.output_fasta}")
    print(f"📈 Using hardcoded Cancer Hotspots mutation frequencies")
    print("=" * 60)
    
    try:
        # Initialize the mutator
        mutator = PeptideMutator()
        
        # Load peptides from input FASTA
        print("Loading peptides from input FASTA...")
        peptides = load_peptides_from_fasta(args.input_fasta)
        print(f"Loaded {len(peptides)} peptides")
        
        # Apply mutations
        mutated_peptides = mutator.mutate_peptide_list(peptides, args.mutations)
        
        # Write to output FASTA
        args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
        write_mutated_fasta(mutated_peptides, args.output_fasta)
        
        print("\n" + "=" * 60)
        print("🎉 Mutation completed successfully!")
        print(f"📁 Mutated peptides saved to: {args.output_fasta}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during mutation: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(mutate_existing_peptides_cli())
