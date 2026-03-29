import matplotlib.pyplot as plt

def plot_missing_overall(df, threshold=0.05):
    miss = df.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]  # only variables with any missingness
    print(len(miss))

    fig, ax = plt.subplots(figsize=(10, len(miss) * 0.35 + 1))
    colors = ['tomato' if v > threshold else 'steelblue' for v in miss]
    ax.barh(miss.index, miss.values, color=colors)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1, label=f'{int(threshold*100)}% threshold')
    ax.set_xlabel('Proportion Missing')
    ax.set_title('Overall Missingness per Variable')
    ax.legend()
    plt.tight_layout()
    plt.savefig('missing_overall.png', dpi=150)
    plt.show()
