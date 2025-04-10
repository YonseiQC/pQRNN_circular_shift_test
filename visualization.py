import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_probs(probs, title="Reg.D Output Probabilities"):
    plt.figure(figsize=(6, 3))
    sns.barplot(x=np.arange(len(probs)), y=np.array(probs))
    plt.xlabel("Class (0 ~ 7)")
    plt.ylabel("Probability")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_hidden_state(h_state, title="Hidden State Amplitude"):
    amplitude = np.abs(np.array(h_state))
    plt.figure(figsize=(6, 3))
    plt.bar(range(len(amplitude)), amplitude)
    plt.title(title)
    plt.xlabel("Basis State Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
