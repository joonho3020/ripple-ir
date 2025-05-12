import matplotlib.pyplot as plt
import numpy as np

def plot_match_percentages_with_src(save_path="match_percentages_with_src.png"):
    # Data
    comparisons = [
        "GCD 1 vs 2",
        "Cordic 1 vs 2",
        "Cordic 2 vs 3",
        "AES 1 vs 2",
        "AES 2 vs 3",
        "FFT 1 vs 2",
        "FFT 2 vs 3",
        "FFT 3 vs 4",
    ]
    match_percentages = [92.73, 17.63, 14.78, 74.39, 99.95, 63.18, 82.35, 18.20]
    src_sizes = [55, 482, 582, 1956, 2147, 220, 340, 1500]

    x = np.arange(len(comparisons))

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Bar plot for match percentages
    bars = ax1.bar(x, match_percentages, color='steelblue')
    ax1.set_ylabel("Match Percentage (%)", color='steelblue')
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparisons, rotation=45, ha="right")
    ax1.tick_params(axis='y', labelcolor='steelblue')

    for bar, pct in zip(bars, match_percentages):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)

    # Secondary axis for Src size (log scale)
    ax2 = ax1.twinx()
    ax2.plot(x, src_sizes, color='darkorange', marker='o', linewidth=2, label="Src Size")
    ax2.set_ylabel("Src Size (log scale)", color='darkorange')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    fig.suptitle("Match Percentage and Source Size Across Comparisons")
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close()
    print(f"Chart saved to {save_path}")

def main():
  plot_match_percentages_with_src()

if __name__ == "__main__":
  main()
