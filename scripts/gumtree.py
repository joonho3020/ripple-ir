import matplotlib.pyplot as plt
import numpy as np

def plot_match_percentages_with_src(save_path="gumtree-results.png"):
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

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar plot for match percentages
    bars = ax1.bar(x, match_percentages, color='steelblue')
    ax1.set_ylabel("Match Percentage (%)", color='steelblue', fontsize=14)
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparisons, rotation=45, ha="right", fontsize=12)
    ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    for bar, pct in zip(bars, match_percentages):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{pct:.1f}%", ha='center', va='bottom', fontsize=11)

    # Secondary axis for Src size (log scale)
    ax2 = ax1.twinx()
    ax2.plot(x, src_sizes, color='darkorange', marker='o', linewidth=2, label="Src Size")
    ax2.set_ylabel("Src Size (log scale)", color='darkorange', fontsize=14)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=12)

    fig.suptitle("Match Percentage and Source Size Across Comparisons", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close()
    print(f"Chart saved to {save_path}")

def plot_size_vs_execution_time(save_path="gumtree-time.png"):
    # Data
    src_sizes = [55, 482, 582, 1956, 2147, 220, 340, 1500]
    exec_times_ms = [
        0.174, 20.471333, 23.857666, 467.543584,
        3.96825, 3.197459, 18.080667, 49.972041
    ]
    labels = [
        "GCD 1 vs 2", "Cordic 1 vs 2", "Cordic 2 vs 3",
        "AES 1 vs 2", "AES 2 vs 3", "FFT 1 vs 2",
        "FFT 2 vs 3", "FFT 3 vs 4"
    ]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(src_sizes, exec_times_ms, color='purple', s=60)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Source Size", fontsize=13)
    plt.ylabel("Execution Time (ms)", fontsize=13)
    plt.title("Execution Time vs. Source Size (log-log)", fontsize=15)

    # Add labels to points
    for x, y, label in zip(src_sizes, exec_times_ms, labels):
        plt.text(x, y, label, fontsize=9, ha='left', va='bottom')

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, format="png")
    plt.close()
    print(f"Size vs execution time plot saved to {save_path}")

def main():
  plot_match_percentages_with_src()
  plot_size_vs_execution_time()

if __name__ == "__main__":
  main()
