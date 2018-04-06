import numpy as np
import matplotlib.pyplot as plt

learning_rates = [0.01, 0.001, 0.005]

plt.figure(figsize=(8,6), dpi=80)
plt.title("Comparison of Loss vs. Epoch for Varied Learning Rates", fontsize=10)
plt.subplot(1,1,1)
plt.xlabel("# of Epochs")
plt.ylabel("Training Loss (Cross-Entropy Loss)")
for r in learning_rates:
    l = np.load("1.1.2_r{}_loss.npy".format(r))
    plt.plot(l, linewidth=1.0, linestyle='-', label="Learning Rate = {}".format(r))
plt.grid('on', linestyle='-', linewidth=0.5)
plt.legend(title="Training Rate", loc="upper right")
plt.savefig("1.1.2_loss_comparison.pdf", format="pdf")

plt.figure(figsize=(8,6), dpi=80)
plt.title("Comparison of Training Set Accuracy vs. Epoch for Varied Learning Rates", fontsize=10)
plt.subplot(1,1,1)
plt.xlabel("# of Epochs")
plt.ylabel("Training Set Accuracy (Percentage Correct)")
for r in learning_rates:
    l = np.load("1.1.2_r{}_train_acc.npy".format(r))
    l *= 100
    plt.plot(l, linewidth=1.0, linestyle='-', label="Learning Rate = {}".format(r))
plt.grid('on', linestyle='-', linewidth=0.5)
plt.legend(title="Training Rate", loc="lower right")
plt.savefig("1.1.2_train_acc_comparison.pdf".format(r), format="pdf")

plt.figure(figsize=(8,6), dpi=80)
plt.title("Comparison of Validation Set Accuracy vs. Epoch for Varied Learning Rates", fontsize=10)
plt.subplot(1,1,1)
plt.xlabel("# of Epochs")
plt.ylabel("Validation Set Accuracy (Percentage Correct)")
for r in learning_rates:
    l = np.load("1.1.2_r{}_valid_acc.npy".format(r))
    l *= 100
    plt.plot(l, linewidth=1.0, linestyle='-', label="Learning Rate = {}".format(r))
plt.grid('on', linestyle='-', linewidth=0.5)
plt.legend(title="Training Rate", loc="lower right")
plt.savefig("1.1.2_valid_acc_comparison.pdf".format(r), format="pdf")

plt.figure(figsize=(8,6), dpi=80)
plt.title("Comparison of Test Set Accuracy vs. Epoch for Varied Learning Rates", fontsize=10)
plt.subplot(1,1,1)
plt.xlabel("# of Epochs")
plt.ylabel("Test Set Accuracy (Percentage Correct)")
for r in learning_rates:
    l = np.load("1.1.2_r{}_test_acc.npy".format(r))
    l *= 100
    plt.plot(l, linewidth=1.0, linestyle='-', label="Learning Rate = {}".format(r))
plt.grid('on', linestyle='-', linewidth=0.5)
plt.legend(title="Training Rate", loc="lower right")
plt.savefig("1.1.2_test_acc_comparison.pdf".format(r), format="pdf")


