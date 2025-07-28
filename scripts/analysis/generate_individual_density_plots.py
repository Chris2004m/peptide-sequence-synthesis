#!/usr/bin/env python3
"""
Generate Individual Density Plots for Each Algorithm × Method Combination
========================================================================

This script generates 30 individual density plots:
- 10 prediction algorithms × 3 peptide generation methods = 30 plots
- Each plot shows the binding score distribution for one specific combination

Author: Generated for peptide benchmarking analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_pvacbind_results():
    """Load pVACbind results from all three methods."""
    
    # Base directory (project root)
    base_dir = Path(__file__).resolve().parents[2]
    # File paths
    random_file = base_dir / "results" / "pvacbind" / "random" / "MHC_Class_I" / "random.all_epitopes.tsv"
    fasta_file = base_dir / "results" / "pvacbind" / "fasta" / "MHC_Class_I" / "fasta.all_epitopes.tsv"
    llm_file = base_dir / "results" / "pvacbind" / "llm" / "MHC_Class_I" / "llm.all_epitopes.tsv"
    
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

def get_all_algorithm_columns(df):
    """Find all algorithm columns and their display names."""
    
    # Look for all columns containing "IC50 Score" or other score patterns
    algorithm_columns = {}
    
    for col in df.columns:
        if 'IC50 Score' in col:
            # Clean up the column name for display
            display_name = col.replace(' IC50 Score', '').replace('_', ' ')
            algorithm_columns[col] = display_name
        elif 'Score' in col and any(alg in col for alg in ['MHC', 'NetMHC', 'SMM', 'Pick']):
            display_name = col.replace(' Score', '').replace('_', ' ')
            algorithm_columns[col] = display_name
    
    print(f"Found {len(algorithm_columns)} algorithm columns:")
    for col, name in algorithm_columns.items():
        print(f"  - {name} ({col})")
    
    return algorithm_columns

def create_individual_density_plots(random_df, fasta_df, llm_df):
    """Create individual density plots for each algorithm-method combination."""
    
    # Combine all datasets
    combined_df = pd.concat([random_df, fasta_df, llm_df], ignore_index=True)
    
    # Get all algorithm columns
    algorithms = get_all_algorithm_columns(combined_df)
    
    # Create output directory
    output_dir = base_dir / "figures" / "individual_density"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating {len(algorithms) * 3} individual density plots...")
    
    plot_count = 0
    methods = ['Random', 'FASTA', 'LLM']
    
    # Create individual plots for each algorithm-method combination
    for col_name, display_name in algorithms.items():
        if col_name not in combined_df.columns:
            print(f"⚠️  Warning: Column '{col_name}' not found, skipping...")
            continue
        
        for method in methods:
            plot_count += 1
            print(f"📊 Creating plot {plot_count}/{len(algorithms)*3}: {display_name} - {method}")
            
            # Get data for this specific method
            method_data = combined_df[combined_df['Method'] == method][col_name]
            method_data = pd.to_numeric(method_data, errors='coerce').dropna()
            
            if len(method_data) == 0:
                print(f"   ⚠️  No data for {display_name} - {method}, skipping...")
                continue
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Log transform IC50 scores for better visualization
            log_data = np.log10(method_data + 1)  # +1 to avoid log(0)
            
            # Create density plot
            sns.histplot(log_data, kde=True, alpha=0.7, color='steelblue',
                        stat='density', bins=50)
            
            # Customize plot
            plt.xlabel(f'Log10({display_name} IC50 Score + 1)')
            plt.ylabel('Density')
            plt.title(f'{display_name} - {method} Peptides\nBinding Score Distribution (n={len(method_data):,})')
            plt.grid(True, alpha=0.3)
            
            # Add statistics text
            median_score = np.median(method_data)
            mean_score = np.mean(method_data)
            std_score = np.std(method_data)
            
            stats_text = f'Median: {median_score:.1f}\nMean: {mean_score:.1f}\nStd: {std_score:.1f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save plot
            safe_alg_name = display_name.replace(' ', '_').replace('/', '_').lower()
            filename = f"{safe_alg_name}_{method.lower()}_density.png"
            filepath = output_dir / filename
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: {filename}")

def create_summary_table(random_df, fasta_df, llm_df):
    """Create a summary table with all statistics."""
    
    combined_df = pd.concat([random_df, fasta_df, llm_df], ignore_index=True)
    algorithms = get_all_algorithm_columns(combined_df)
    
    print("\n📊 Generating comprehensive statistics table...")
    
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
                    'Max': np.max(method_data),
                    'Q25': np.percentile(method_data, 25),
                    'Q75': np.percentile(method_data, 75)
                })
    
    # Create DataFrame and save
    stats_df = pd.DataFrame(stats_data)
    output_dir = base_dir / "figures" / "individual_density"
    stats_path = base_dir / "statistics" / "comprehensive_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    
    print(f"   ✅ Saved comprehensive statistics: {stats_path}")
    
    return stats_df

def main():
    """Main analysis function."""
    
    print("🧬 Individual Density Plots Generation")
    print("="*50)
    
    # Load data
    random_df, fasta_df, llm_df = load_pvacbind_results()
    
    if random_df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Create individual density plots
    create_individual_density_plots(random_df, fasta_df, llm_df)
    
    # Create comprehensive statistics
    stats_df = create_summary_table(random_df, fasta_df, llm_df)
    
    # Count total plots generated
    output_dir = base_dir / "figures" / "individual_density"
    plot_files = list(output_dir.glob("*_density.png"))
    
    print("\n" + "="*60)
    print("✅ INDIVIDUAL DENSITY PLOTS COMPLETE!")
    print(f"📁 Generated {len(plot_files)} individual density plots")
    print(f"📊 All plots saved in: figures/individual_density/")
    print(f"📈 Comprehensive statistics saved")
    print("="*60)
    
    # Print summary by algorithm
    print("\nSUMMARY BY ALGORITHM:")
    for algorithm in stats_df['Algorithm'].unique():
        print(f"\n{algorithm}:")
        alg_data = stats_df[stats_df['Algorithm'] == algorithm]
        for _, row in alg_data.iterrows():
            print(f"  {row['Method']:8s}: {row['Count']:5d} peptides, Median = {row['Median']:8.1f}")

if __name__ == "__main__":
    main()
