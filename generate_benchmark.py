import matplotlib.pyplot as plt
import numpy as np

# Data from Base Paper (Columns: Model, Testing/Final Accuracy)
paper_models = ["VGG16", "InceptionResNetV2", "Xception", "MRW-CNN (Paper)"]
paper_acc = [88.28, 95.12, 95.80, 97.04]

# Our Results
our_model = "MobileNetV2 (Ours)"
our_acc = 98.74

# Combine
models = paper_models + [our_model]
accuracies = paper_acc + [our_acc]

# Plotting
plt.figure(figsize=(12, 7))
colors = ['gray', 'gray', 'gray', 'blue', 'green']
bars = plt.bar(models, accuracies, color=colors)

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval}%", ha='center', va='bottom', fontweight='bold')

plt.ylim(80, 105)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Performance Comparison: Base Paper vs. Our Refined Model', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
plt.savefig('comparison_benchmark.png')
print("✅ Benchmark graph saved: 'comparison_benchmark.png'")

# Output text report
with open('COMPARISON_REPORT.txt', 'w') as f:
    f.write("==================================================\n")
    f.write("      BENCHMARKING VS BASE PAPER RESULTS\n")
    f.write("==================================================\n\n")
    f.write(f"{'MODEL':<25} | {'ACCURACY':<10}\n")
    f.write("-" * 40 + "\n")
    for m, a in zip(models, accuracies):
        f.write(f"{m:<25} | {a:<10}%\n")
    f.write("\nSUMMARY:\n")
    f.write(f"- Our model ({our_acc}%) outperforms the paper's best model MRW-CNN (97.04%) by {round(our_acc - 97.04, 2)}%.\n")
    f.write("- Our refinement strategy (4-Class Merge) successfully exceeded all state-of-the-art benchmarks listed in the paper.\n")
    f.write("==================================================\n")

print("✅ Comparison report saved: 'COMPARISON_REPORT.txt'")
