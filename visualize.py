import matplotlib.pyplot as plt
import numpy as np
import json

def plot_unlearning_results():
    """Visualize unlearning performance"""
    # This would load results from your experiments
    groups = [1, 3, 5, 7, 10]
    accuracies_before = [0.82, 0.81, 0.80, 0.79, 0.78]
    accuracies_after = [0.75, 0.73, 0.71, 0.70, 0.68]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies_before, width, label='Before Unlearning')
    bars2 = ax.bar(x + width/2, accuracies_after, width, label='After Unlearning')
    
    ax.set_xlabel('Number of Groups')
    ax.set_ylabel('Accuracy')
    ax.set_title('UltraRE Unlearning Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/unlearning_performance.png')
    plt.show()

if __name__ == "__main__":
    plot_unlearning_results()