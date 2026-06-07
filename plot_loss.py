import matplotlib.pyplot as plt

epochs = list(range(1, 51))
losses = [
    3.0156, 2.5306, 2.2994, 2.1873, 2.1052,
    2.0352, 1.9795, 1.9261, 1.8775, 1.8352,
    1.7913, 1.7483, 1.7098, 1.6728, 1.6369,
    1.6037, 1.5682, 1.5381, 1.5066, 1.4743,
    1.4455, 1.4148, 1.3862, 1.3601, 1.3333,
    1.3066, 1.2818, 1.2550, 1.2316, 1.2047,
    1.1817, 1.1571, 1.1329, 1.1134, 1.0884,
    1.0616, 1.0413, 1.0184, 0.9951, 0.9728,
    0.9554, 0.9307, 0.9116, 0.8905, 0.8727,
    0.8521, 0.8381, 0.8279, 0.7960, 0.7748
]

plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, color='#004b5a', linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Cross-Entropy Loss', fontsize=14)
plt.title('LSTM Training Loss — Tolkien Character-Level Text Generation', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=150)
plt.show()
print("Saved to training_loss.png")
