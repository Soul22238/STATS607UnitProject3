import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_file='results/raw/simulation_results_nsim20000.csv'):
    """
    Plot average power vs m for different methods.
    Create two figures for different L values.
    Each figure has 4x3 subplots: rows = null_ratio (0.75, 0.5, 0.25, 0.0), columns = mode (D, E, I)
    """
    df = pd.read_csv(csv_file)
    
    null_ratios = [0.75, 0.5, 0.25, 0.0]
    modes = ['D', 'E', 'I']
    corrections = ['bonferroni', 'hochberg', 'fdr']
    
    # Plot for each L value
    for L in [5, 8, 10, 15]:
        fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True, sharey=True)
        fig.suptitle(f'Average Power vs m (L={L})', fontsize=14, y=0.995)
        
        for row, null_ratio in enumerate(null_ratios):
            for col, mode in enumerate(modes):
                ax = axes[row, col]
                
                # Filter data for this L, null_ratio, and mode
                mask = (df['L'] == L) & (df['null_ratio'] == null_ratio) & (df['mode'] == mode)
                subset = df[mask]
                
                # Plot each correction method
                for correction in corrections:
                    data = subset[subset['correction'] == correction].sort_values('m')
                    ax.plot(data['m'], data['power'], marker='o', label=correction)
                
                # Labels
                if row == 0:
                    ax.set_title(f'Mode {mode}')
                if col == 0:
                    ax.set_ylabel(f'π₀={null_ratio}\nPower')
                if row == 3:
                    ax.set_xlabel('m')
                
                ax.grid(True, alpha=0.3)
        
        # Add legend to the right side of the figure
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5), fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        output_file = f'results/figures/power_comparison_L{L}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_file}")
        plt.close()

if __name__ == "__main__":
    plot_results()
