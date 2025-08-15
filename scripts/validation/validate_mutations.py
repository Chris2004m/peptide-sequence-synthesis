#!/usr/bin/env python3
"""
Validation script for peptide mutation functionality.

This script tests:
1. Position randomness - mutations occur at random positions
2. Amino acid substitution weights match empirical cancer data
3. No silent mutations occur
4. Statistical significance of distributions
"""

import sys
import os
import numpy as np
from collections import defaultdict, Counter
from scipy import stats

# Add the generation scripts to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generation'))

from peptide_mutations import PeptideMutator


class MutationValidator:
    def __init__(self, num_tests=10000):
        """Initialize validator with specified number of test mutations."""
        self.num_tests = num_tests
        self.mutator = PeptideMutator()
        self.results = {
            'position_counts': defaultdict(int),
            'substitution_counts': defaultdict(lambda: defaultdict(int)),
            'silent_mutations': 0,
            'total_mutations': 0
        }
    
    def validate_position_randomness(self, peptide_length=9):
        """Test that mutation positions are uniformly random."""
        print(f"🧪 Testing position randomness with {self.num_tests} mutations on {peptide_length}-mers...")
        
        # Create identical test peptides
        test_peptide = "A" * peptide_length
        
        for i in range(self.num_tests):
            # Get the mutated peptide and find the position that changed
            mutated = self.mutator.mutate_peptide(test_peptide, num_mutations=1)
            
            # Find the position that was mutated
            for pos in range(peptide_length):
                if test_peptide[pos] != mutated[pos]:
                    self.results['position_counts'][pos] += 1
                    break
        
        # Analyze uniformity
        positions = list(range(peptide_length))
        observed_counts = [self.results['position_counts'][pos] for pos in positions]
        expected_count = self.num_tests / peptide_length
        
        # Chi-square test for uniformity
        chi2_stat, p_value = stats.chisquare(observed_counts)
        
        print(f"📊 Position distribution:")
        for pos, count in enumerate(observed_counts):
            print(f"  Position {pos}: {count} ({count/self.num_tests*100:.1f}%)")
        
        print(f"📈 Chi-square test for uniformity:")
        print(f"  Chi-square statistic: {chi2_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  {'✅ PASS' if p_value > 0.05 else '❌ FAIL'} - Positions appear {'uniform' if p_value > 0.05 else 'non-uniform'}")
        
        return p_value > 0.05
    
    def validate_substitution_weights(self, source_aa='A', num_tests=50000):
        """Test that amino acid substitutions match empirical probabilities."""
        print(f"🧪 Testing substitution weights from {source_aa} with {num_tests} mutations...")
        
        # Create test peptides with the source amino acid
        test_peptide = source_aa * 5  # 5-mer for simplicity
        
        substitution_counts = Counter()
        
        for i in range(num_tests):
            mutated = self.mutator.mutate_peptide(test_peptide, num_mutations=1)
            
            # Find the amino acid that it was changed to
            for pos in range(len(test_peptide)):
                if test_peptide[pos] != mutated[pos]:
                    target_aa = mutated[pos]
                    substitution_counts[target_aa] += 1
                    break
        
        # Get expected probabilities from the mutator
        if source_aa in self.mutator.substitution_probabilities:
            # Build expected_probs dictionary from targets and probabilities lists
            sub_data = self.mutator.substitution_probabilities[source_aa]
            expected_probs = {}
            for target_aa, prob in zip(sub_data['targets'], sub_data['probabilities']):
                expected_probs[target_aa] = prob
            
            # Calculate observed frequencies
            total_observed = sum(substitution_counts.values())
            observed_freqs = {aa: count / total_observed for aa, count in substitution_counts.items()}
            
            print(f"📊 Substitution analysis for {source_aa}:")
            print(f"{'Target AA':<10} {'Expected':<10} {'Observed':<10} {'Diff':<10}")
            print("-" * 45)
            
            chi2_components = []
            for target_aa in sorted(expected_probs.keys()):
                expected_freq = expected_probs[target_aa]
                observed_freq = observed_freqs.get(target_aa, 0)
                expected_count = expected_freq * total_observed
                observed_count = substitution_counts.get(target_aa, 0)
                
                diff = abs(expected_freq - observed_freq)
                print(f"{target_aa:<10} {expected_freq:<10.4f} {observed_freq:<10.4f} {diff:<10.4f}")
                
                # Contribution to chi-square
                if expected_count > 0:
                    chi2_components.append((observed_count - expected_count) ** 2 / expected_count)
            
            # Chi-square goodness of fit test
            chi2_stat = sum(chi2_components)
            degrees_of_freedom = len(expected_probs) - 1
            p_value = 1 - stats.chi2.cdf(chi2_stat, degrees_of_freedom)
            
            print(f"\n📈 Chi-square goodness-of-fit test:")
            print(f"  Chi-square statistic: {chi2_stat:.4f}")
            print(f"  Degrees of freedom: {degrees_of_freedom}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  {'✅ PASS' if p_value > 0.05 else '❌ FAIL'} - Substitution frequencies {'match' if p_value > 0.05 else 'do not match'} expected probabilities")
            
            return p_value > 0.05
        else:
            print(f"❌ No substitution probabilities found for {source_aa}")
            return False
    
    def validate_no_silent_mutations(self, num_tests=1000):
        """Test that no silent mutations occur."""
        print(f"🧪 Testing for silent mutations with {num_tests} mutations...")
        
        silent_count = 0
        test_peptides = [
            "ACDEFGHIK",  # 9-mer
            "LMNPQRST",   # 8-mer  
            "VWYA"        # 4-mer
        ]
        
        for peptide in test_peptides:
            for i in range(num_tests // len(test_peptides)):
                mutated = self.mutator.mutate_peptide(peptide, num_mutations=1)
                if peptide == mutated:
                    silent_count += 1
        
        print(f"📊 Silent mutation analysis:")
        print(f"  Silent mutations: {silent_count}/{num_tests}")
        print(f"  {'✅ PASS' if silent_count == 0 else '❌ FAIL'} - {'No silent mutations detected' if silent_count == 0 else f'{silent_count} silent mutations found'}")
        
        return silent_count == 0
    
    def validate_multiple_mutations(self, num_tests=1000):
        """Test that multiple mutations work correctly."""
        print(f"🧪 Testing multiple mutations with {num_tests} tests...")
        
        test_peptide = "ACDEFGHIK"  # 9-mer
        mutation_counts = [1, 2, 3, 5, 9]  # Test different numbers of mutations
        
        for num_mutations in mutation_counts:
            valid_count = 0
            for i in range(num_tests // len(mutation_counts)):
                mutated = self.mutator.mutate_peptide(test_peptide, num_mutations=num_mutations)
                
                # Count actual differences
                actual_mutations = sum(1 for orig, mut in zip(test_peptide, mutated) if orig != mut)
                
                if actual_mutations == num_mutations:
                    valid_count += 1
            
            success_rate = valid_count / (num_tests // len(mutation_counts))
            print(f"  {num_mutations} mutations: {success_rate*100:.1f}% correct")
    
    def run_all_validations(self):
        """Run all validation tests."""
        print("🧬 Starting Peptide Mutation Validation")
        print("=" * 50)
        
        results = {}
        
        # Test 1: Position randomness
        results['position_randomness'] = self.validate_position_randomness()
        print()
        
        # Test 2: Substitution weights for common amino acids
        common_aas = ['A', 'L', 'S', 'G', 'V']  # Test a few common amino acids
        substitution_results = []
        for aa in common_aas:
            if aa in self.mutator.substitution_probabilities:
                result = self.validate_substitution_weights(aa, num_tests=10000)
                substitution_results.append(result)
                print()
        
        results['substitution_weights'] = all(substitution_results) if substitution_results else False
        
        # Test 3: No silent mutations
        results['no_silent_mutations'] = self.validate_no_silent_mutations()
        print()
        
        # Test 4: Multiple mutations
        self.validate_multiple_mutations()
        print()
        
        # Summary
        print("📋 VALIDATION SUMMARY")
        print("=" * 30)
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
        
        overall_pass = all(results.values())
        print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}")
        
        return overall_pass


def main():
    """Main function to run validation."""
    print("🧬 Peptide Mutation System Validation")
    print("====================================")
    
    validator = MutationValidator(num_tests=10000)
    success = validator.run_all_validations()
    
    if success:
        print("\n🎉 All validation tests passed! The mutation system is working correctly.")
    else:
        print("\n⚠️  Some validation tests failed. Please review the results above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
