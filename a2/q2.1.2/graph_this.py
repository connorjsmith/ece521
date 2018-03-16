import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
a_v = np.load("Adam Optimizer_v_acc.npy")
a_t = np.load("Adam Optimizer_t_acc.npy")
g_v = np.load("Gradient Descent_v_acc.npy")
g_t = np.load("Gradient Descent_t_acc.npy")
plt.plot(g_v, linewidth=1.0, linestyle='-', label="SGD Validation Data Accuracy")
plt.plot(g_t, linewidth=1.0, linestyle='-', label="SGD Training Data Accuracy")
plt.plot(a_v, linewidth=1.0, linestyle='-', label="Adam Optimizer Validation Data Accuracy")
plt.plot(a_t, linewidth=1.0, linestyle='-', label="Adam Optimizer Training Data Accuracy")
plt.title("Adam Optimizer vs SGD with Learning Rate η = 0.001", fontsize=10)
plt.xlabel("# of Epochs")
plt.ylabel("Accuracy (Percentage Correct)")
plt.legend(title="Data Set", loc="lower right")
plt.grid('on', linestyle='-', linewidth=0.5)
plt.axhline(linewidth=2, color='r', y=1)
plt.savefig("accuracy_comparison.pdf", format="pdf")




plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
a_l = np.load("Adam Optimizer_loss.npy")
g_l = np.load("Gradient Descent_loss.npy")
plt.plot(g_l, linewidth=1.0, linestyle='-', label="SGD Cross-Entropy Loss")
plt.plot(a_l, linewidth=1.0, linestyle='-', label="Adam Optimizer Cross-Entropy Loss")
plt.title("Adam Optimizer vs SGD with Learning Rate η = 0.001", fontsize=10)
plt.xlabel("# of Epochs")
plt.ylabel("Training Loss (Cross-Entropy Loss)")
plt.legend(title="Training Rate", loc="upper right")
plt.grid('on', linestyle='-', linewidth=0.5)
plt.savefig("loss_comparison.pdf", format="pdf")
