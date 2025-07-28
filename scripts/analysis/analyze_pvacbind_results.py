#!/usr/bin/env python3
"""
Analyze pVACbind Results and Generate Density Plots
==================================================

This script analyzes the binding score distributions from three peptide generation methods:
1. Random peptides
2. FASTA-sampled peptides  
3. LLM-generated peptides

For each of the 10 prediction algorithms, it creates density plots comparing the three methods.

Author: Generated for peptide benchmarking analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Project base directory
BASE_DIR = Path(__file__).resolve().parents[2]
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_pvacbind_results():
    """Load pVACbind results from all three methods."""
    
    # Base directory (project root)
    
    # File paths
    random_file = BASE_DIR / "results" / "pvacbind" / "random" / "MHC_Class_I" / "random.all_epitopes.tsv"
    fasta_file = BASE_DIR / "results" / "pvacbind" / "fasta" / "MHC_Class_I" / "fasta.all_epitopes.tsv"
    llm_file = BASE_DIR / "results" / "pvacbind" / "llm" / "MHC_Class_I" / "llm.all_epitopes.tsv"
    
    # Load datasets
    print("Loading pVACbind results...")
    try:
        random_df = pd.read_csv(random_file, sep='\t')
        fasta_df = pd.read_csv(fasta_file, sep='\t')
        llm_df = pd.read_csv(llm_file, sep='\t')
        
        # Add method column
        random_df['Method'] = 'Random'
        fasta_df['Method'] = 'FASTA'
        llm_df['Method'] = 'LLM'
        
        print(f"✅ Random peptides: {len(random_df):,} entries")
        print(f"✅ FASTA peptides: {len(fasta_df):,} entries")
        print(f"✅ LLM peptides: {len(llm_df):,} entries")
        
        return random_df, fasta_df, llm_df
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None

def get_algorithm_columns():
    """Define the algorithm columns and their display names."""
    
    # Algorithm columns in the TSV files (IC50 scores)
    algorithms = {
        'MHCflurry IC50 Score': 'MHCflurry',
        'MHCnuggetsI IC50 Score': 'MHCnuggetsI', 
        'NetMHCcons IC50 Score': 'NetMHCcons',
        'NetMHCpan IC50 Score': 'NetMHCpan',
        'PickPocket IC50 Score': 'PickPocket',
        'SMM IC50 Score': 'SMM',
        'SMMPMBEC IC50 Score': 'SMMPMBEC'
    }
    
    return algorithms

def create_density_plots(random_df, fasta_df, llm_df):
    """Create density plots for each algorithm comparing the three methods."""
    
    # Combine all datasets
    combined_df = pd.concat([random_df, fasta_df, llm_df], ignore_index=True)
    
    # Get algorithm columns
    algorithms = get_algorithm_columns()
    
    # Create output directory
    output_dir = BASE_DIR / "figures" / "algorithm_density"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating density plots for {len(algorithms)} algorithms...")
    
    # Create individual plots for each algorithm
    for col_name, display_name in algorithms.items():
        if col_name not in combined_df.columns:
            print(f"⚠️  Warning: Column '{col_name}' not found, skipping...")
            continue
            
        print(f"📊 Creating density plot for {display_name}...")
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create density plot for each method
        for method in ['Random', 'FASTA', 'LLM']:
            method_data = combined_df[combined_df['Method'] == method][col_name]
            
            # Remove NaN values and convert to numeric
            method_data = pd.to_numeric(method_data, errors='coerce').dropna()
            
            if len(method_data) > 0:
                # Log transform IC50 scores for better visualization
                log_data = np.log10(method_data + 1)  # +1 to avoid log(0)
                
                # Create density plot
                sns.histplot(log_data, kde=True, alpha=0.6, label=f'{method} (n={len(method_data):,})',
                           stat='density', bins=50)
        
        # Customize plot
        plt.xlabel(f'Log10({display_name} IC50 Score + 1)')
        plt.ylabel('Density')
        plt.title(f'Binding Score Distribution: {display_name}\nComparison Across Peptide Generation Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = []
        for method in ['Random', 'FASTA', 'LLM']:
            method_data = combined_df[combined_df['Method'] == method][col_name]
            method_data = pd.to_numeric(method_data, errors='coerce').dropna()
            if len(method_data) > 0:
                median_score = np.median(method_data)
                stats_text.append(f'{method}: Median = {median_score:.1f}')
        
        plt.text(0.02, 0.98, '\n'.join(stats_text), transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        filename = f"{display_name.replace(' ', '_').lower()}_density_plot.png"
        filepath = output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {filepath}")

def create_summary_plot(random_df, fasta_df, llm_df):
    """Create a summary plot showing all algorithms together."""
    
    # Combine datasets
    combined_df = pd.concat([random_df, fasta_df, llm_df], ignore_index=True)
    algorithms = get_algorithm_columns()
    
    print("\n📊 Creating summary comparison plot...")
    
    # Create subplots
    n_algorithms = len(algorithms)
    cols = 3
    rows = (n_algorithms + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
    axes = axes.flatten() if n_algorithms > 1 else [axes]
    
    for idx, (col_name, display_name) in enumerate(algorithms.items()):
        if col_name not in combined_df.columns:
            continue
            
        ax = axes[idx]
        
        # Plot density for each method
        for method in ['Random', 'FASTA', 'LLM']:
            method_data = combined_df[combined_df['Method'] == method][col_name]
            method_data = pd.to_numeric(method_data, errors='coerce').dropna()
            
            if len(method_data) > 0:
                log_data = np.log10(method_data + 1)
                sns.histplot(log_data, kde=True, alpha=0.6, label=method,
                           stat='density', bins=30, ax=ax)
        
        ax.set_xlabel(f'Log10(IC50 + 1)')
        ax.set_ylabel('Density')
        ax.set_title(f'{display_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(algorithms), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Binding Score Distributions: All Algorithms\nComparison Across Peptide Generation Methods', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save summary plot
    output_dir = BASE_DIR / "figures" / "algorithm_density"
    summary_path = output_dir / "summary_all_algorithms.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved summary plot: {summary_path}")

def generate_statistics_report(random_df, fasta_df, llm_df):
    """Generate a statistical summary report."""
    
    combined_df = pd.concat([random_df, fasta_df, llm_df], ignore_index=True)
    algorithms = get_algorithm_columns()
    
    print("\n📊 Generating statistical summary...")
    
    # Create statistics table
    stats_data = []
    
    for col_name, display_name in algorithms.items():
        if col_name not in combined_df.columns:
            continue
            
        for method in ['Random', 'FASTA', 'LLM']:
            method_data = combined_df[combined_df['Method'] == method][col_name]
            method_data = pd.to_numeric(method_data, errors='coerce').dropna()
            
            if len(method_data) > 0:
                stats_data.append({
                    'Algorithm': display_name,
                    'Method': method,
                    'Count': len(method_data),
                    'Mean': np.mean(method_data),
                    'Median': np.median(method_data),
                    'Std': np.std(method_data),
                    'Min': np.min(method_data),
                    'Max': np.max(method_data)
                })
    
    # Create DataFrame and save
    stats_df = pd.DataFrame(stats_data)
    output_dir = BASE_DIR / "figures" / "algorithm_density"
    stats_path = output_dir / "binding_score_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    
    print(f"   ✅ Saved statistics: {stats_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BINDING SCORE ANALYSIS SUMMARY")
    print("="*60)
    for algorithm in stats_df['Algorithm'].unique():
        print(f"\n{algorithm}:")
        alg_data = stats_df[stats_df['Algorithm'] == algorithm]
        for _, row in alg_data.iterrows():
            print(f"  {row['Method']:8s}: Median = {row['Median']:8.1f}, Mean = {row['Mean']:8.1f}")

def main():
    """Main analysis function."""
    
    print("🧬 pVACbind Results Analysis")
    print("="*50)
    
    # Load data
    random_df, fasta_df, llm_df = load_pvacbind_results()
    
    if random_df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Create density plots
    create_density_plots(random_df, fasta_df, llm_df)
    
    # Create summary plot
    create_summary_plot(random_df, fasta_df, llm_df)
    
    # Generate statistics
    generate_statistics_report(random_df, fasta_df, llm_df)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("📁 All plots saved in: figures/algorithm_density/")
    print("📊 Individual density plots for each algorithm")
    print("📊 Summary plot with all algorithms")
    print("📈 Statistical summary CSV file")
    print("="*60)

if __name__ == "__main__":
    main()
