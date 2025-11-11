"""
Visualize Grid Search Results

This script creates visualizations to analyze grid search results and identify
optimal hyperparameter combinations.

Usage:
    python visualize_grid_search.py grid_search_results.csv
    python visualize_grid_search.py quick_grid_search_results.csv --output quick_analysis.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_results(csv_path):
    """Load grid search results from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} parameter combinations from {csv_path}")
    return df


def plot_top_configurations(df, n=10):
    """Plot top N configurations by sequence accuracy."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    top_n = df.nlargest(n, 'sequence_accuracy').copy()
    top_n['config'] = [
        f"swx={row['spatial_weight_x']:.1f}\nswy={row['spatial_weight_y']:.1f}\ncw={row['contour_weight']:.0f}"
        for _, row in top_n.iterrows()
    ]
    
    x = np.arange(len(top_n))
    width = 0.35
    
    ax.bar(x - width/2, top_n['sequence_accuracy'], width, label='Sequence Acc', alpha=0.8)
    ax.bar(x + width/2, top_n['character_accuracy'], width, label='Character Acc', alpha=0.8)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Top {n} Hyperparameter Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(top_n['config'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_parameter_effects(df):
    """Plot the effect of each parameter on accuracy."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    params = ['spatial_weight_x', 'spatial_weight_y', 'contour_weight']
    metrics = ['sequence_accuracy', 'character_accuracy']
    
    for j, metric in enumerate(metrics):
        for i, param in enumerate(params):
            ax = axes[j, i]
            
            # Group by parameter and compute mean and std
            grouped = df.groupby(param)[metric].agg(['mean', 'std', 'count'])
            
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       marker='o', linestyle='-', capsize=5, capthick=2, linewidth=2)
            
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)")
            ax.set_title(f"Effect of {param.replace('_', ' ')} on {metric.replace('_', ' ')}")
            ax.grid(True, alpha=0.3)
            
            # Add point labels showing count
            for x, (mean, count) in zip(grouped.index, zip(grouped['mean'], grouped['count'])):
                ax.annotate(f'n={int(count)}', (x, mean), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_heatmaps(df):
    """Create heatmaps showing interactions between parameters."""
    unique_swx = sorted(df['spatial_weight_x'].unique())
    unique_swy = sorted(df['spatial_weight_y'].unique())
    unique_cw = sorted(df['contour_weight'].unique())
    
    # We'll create 3 heatmaps: swx vs swy, swx vs cw, swy vs cw
    # Each averaged over the third parameter
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. spatial_weight_x vs spatial_weight_y (averaged over contour_weight)
    pivot1 = df.groupby(['spatial_weight_x', 'spatial_weight_y'])['sequence_accuracy'].mean().unstack()
    sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Sequence Acc (%)'})
    axes[0].set_title('Spatial Weight X vs Y\n(averaged over contour_weight)')
    axes[0].set_xlabel('spatial_weight_y')
    axes[0].set_ylabel('spatial_weight_x')
    
    # 2. spatial_weight_x vs contour_weight (averaged over spatial_weight_y)
    pivot2 = df.groupby(['spatial_weight_x', 'contour_weight'])['sequence_accuracy'].mean().unstack()
    sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Sequence Acc (%)'})
    axes[1].set_title('Spatial Weight X vs Contour Weight\n(averaged over spatial_weight_y)')
    axes[1].set_xlabel('contour_weight')
    axes[1].set_ylabel('spatial_weight_x')
    
    # 3. spatial_weight_y vs contour_weight (averaged over spatial_weight_x)
    pivot3 = df.groupby(['spatial_weight_y', 'contour_weight'])['sequence_accuracy'].mean().unstack()
    sns.heatmap(pivot3, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[2], cbar_kws={'label': 'Sequence Acc (%)'})
    axes[2].set_title('Spatial Weight Y vs Contour Weight\n(averaged over spatial_weight_x)')
    axes[2].set_xlabel('contour_weight')
    axes[2].set_ylabel('spatial_weight_y')
    
    plt.tight_layout()
    return fig


def plot_scatter_matrix(df):
    """Create scatter plot matrix showing relationships."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    params = ['spatial_weight_x', 'spatial_weight_y', 'contour_weight']
    
    for i, param1 in enumerate(params):
        for j, param2 in enumerate(params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(df[param1], bins=20, alpha=0.7, edgecolor='black')
                ax.set_xlabel(param1.replace('_', ' ').title())
                ax.set_ylabel('Count')
            else:
                # Off-diagonal: scatter plot
                scatter = ax.scatter(df[param2], df[param1], 
                                    c=df['sequence_accuracy'], 
                                    cmap='YlOrRd', s=50, alpha=0.6)
                ax.set_xlabel(param2.replace('_', ' ').title() if i == 2 else '')
                ax.set_ylabel(param1.replace('_', ' ').title() if j == 0 else '')
                
                if i == 0 and j == 2:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Sequence Acc (%)', rotation=270, labelpad=20)
    
    plt.suptitle('Parameter Scatter Matrix (colored by Sequence Accuracy)', y=0.995, fontsize=14)
    plt.tight_layout()
    return fig


def print_summary_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("GRID SEARCH SUMMARY")
    print("="*70)
    
    print(f"\nTotal configurations tested: {len(df)}")
    print(f"Total test images: {df['total_images'].iloc[0]}")
    print(f"Total time: {df['time_seconds'].sum():.1f} seconds ({df['time_seconds'].sum()/60:.1f} minutes)")
    
    print("\n" + "-"*70)
    print("BEST CONFIGURATION")
    print("-"*70)
    best_row = df.loc[df['sequence_accuracy'].idxmax()]
    print(f"  spatial_weight_x: {best_row['spatial_weight_x']}")
    print(f"  spatial_weight_y: {best_row['spatial_weight_y']}")
    print(f"  contour_weight: {best_row['contour_weight']}")
    print(f"  Sequence Accuracy: {best_row['sequence_accuracy']:.2f}%")
    print(f"  Character Accuracy: {best_row['character_accuracy']:.2f}%")
    
    print("\n" + "-"*70)
    print("TOP 5 CONFIGURATIONS")
    print("-"*70)
    top5 = df.nlargest(5, 'sequence_accuracy')
    for idx, row in top5.iterrows():
        print(f"\n  Rank {list(top5.index).index(idx) + 1}:")
        print(f"    swx={row['spatial_weight_x']}, swy={row['spatial_weight_y']}, cw={row['contour_weight']}")
        print(f"    Seq Acc: {row['sequence_accuracy']:.2f}%, Char Acc: {row['character_accuracy']:.2f}%")
    
    print("\n" + "-"*70)
    print("PARAMETER STATISTICS")
    print("-"*70)
    
    for param in ['spatial_weight_x', 'spatial_weight_y', 'contour_weight']:
        print(f"\n  {param}:")
        param_stats = df.groupby(param)['sequence_accuracy'].agg(['mean', 'std', 'min', 'max'])
        print(f"    Values tested: {sorted(df[param].unique())}")
        best_val = param_stats['mean'].idxmax()
        print(f"    Best value (by mean): {best_val} (mean acc: {param_stats.loc[best_val, 'mean']:.2f}%)")
    
    print("\n" + "="*70 + "\n")


def generate_report(csv_path, output_dir='grid_search_analysis'):
    """Generate complete analysis report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Analyzing results from {csv_path}...")
    df = load_results(csv_path)
    
    # Print statistics
    print_summary_statistics(df)
    
    # Generate plots
    print("Generating visualizations...")
    
    print("  1. Top configurations...")
    fig1 = plot_top_configurations(df, n=10)
    fig1.savefig(output_dir / 'top_configurations.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("  2. Parameter effects...")
    fig2 = plot_parameter_effects(df)
    fig2.savefig(output_dir / 'parameter_effects.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("  3. Parameter interaction heatmaps...")
    fig3 = plot_heatmaps(df)
    fig3.savefig(output_dir / 'parameter_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("  4. Scatter matrix...")
    fig4 = plot_scatter_matrix(df)
    fig4.savefig(output_dir / 'scatter_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    print("  - top_configurations.png")
    print("  - parameter_effects.png")
    print("  - parameter_heatmaps.png")
    print("  - scatter_matrix.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize grid search results')
    parser.add_argument('csv_path', type=str, help='Path to grid search results CSV')
    parser.add_argument('--output', type=str, default='grid_search_analysis',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    generate_report(args.csv_path, args.output)


if __name__ == '__main__':
    main()
