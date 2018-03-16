import numpy as np
import matplotlib.pyplot as plt

learning_rates = [1, 0.1, 0.01, 0.0001, 0.005, 0.001]
for r in learning_rates:
    plt.figure(figsize=(8,6), dpi=80)
    plt.subplot(1,1,1)
    l = np.load("log_q1_{}_loss.npy".format(r))
    plt.plot(l, linewidth=1.0, linestyle='-', label="Cross-Entropy Loss")
    plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500, Training Rate η = "+str(r), fontsize=10)
    plt.xlabel("# of Epochs")
    plt.ylabel("Training Loss (Cross-Entropy Loss)")
    plt.legend(title="Training Rate", loc="upper right")
    plt.grid('on', linestyle='-', linewidth=0.5)
    plt.savefig("learning_{}_loss.pdf".format(r), format="pdf")

    plt.figure(figsize=(8,6), dpi=80)
    plt.subplot(1,1,1)
    v = np.load("log_q1_{}_valid_acc.npy".format(r))
    t = np.load("log_q1_{}_train_acc.npy".format(r))
    plt.plot(v, linewidth=1.0, linestyle='-', label="Validation Data Accuracy")
    plt.plot(t, linewidth=1.0, linestyle='-', label="Training Data Accuracy")
    plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500, Training Rate η = "+str(r), fontsize=10)
    plt.xlabel("# of Epochs")
    plt.ylabel("Accuracy (Percentage Correct)")
    plt.axhline(linewidth=2, color='r', y=1)
    plt.legend(title="Data Set", loc="lower right")
    plt.grid('on', linestyle='-', linewidth=0.5)
    plt.savefig("learning_{}_accuracy.pdf".format(r), format="pdf")

plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
l01 = np.load("log_q1_0.1_loss.npy")
l001 = np.load("log_q1_0.1_loss.npy")
l1 = np.load("log_q1_1_loss.npy")
l0001 = np.load("log_q1_0.001_loss.npy")
l0005 = np.load("log_q1_0.005_loss.npy")
l00001 = np.load("log_q1_0.0001_loss.npy")
plt.plot(l1, linewidth=1.0, linestyle='-', label="1.0")
plt.plot(l01, linewidth=1.0, linestyle='-', label="0.1")
plt.plot(l001, linewidth=1.0, linestyle='-', label="0.01")
plt.plot(l0001, linewidth=1.0, linestyle='-', label="0.001")
plt.plot(l0005, linewidth=1.0, linestyle='-', label="0.005")
plt.plot(l00001, linewidth=1.0, linestyle='-', label="0.0001")
plt.xlabel("# of Epochs")
plt.ylabel("Training Loss (Cross-Entropy Loss)")
plt.legend(title="Training Rate ()", loc="upper right")
plt.grid('on', linestyle='-', linewidth=0.5)
plt.savefig("learning_comparison_loss.pdf", format="pdf")
