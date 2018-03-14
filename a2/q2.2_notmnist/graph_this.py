import numpy as np
import matplotlib.pyplot as plt

learning_rates = [0.005, 0.0025, 0.001, 0.0005, 0.0001]
plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
for r in learning_rates:
    l = np.load("2.2.1_{}_notmnist_loss.npy".format(r))
    plt.plot(l, linewidth=1.0, linestyle='-', label=("Learning Rate = {} Loss".format(r)))
plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500", fontsize=10)
plt.xlabel("# of Epochs")
plt.ylabel("Training Loss (Cross-Entropy Loss)")
plt.legend(title="Training Rate", loc="upper right")
plt.grid('on', linestyle='-', linewidth=0.5)
plt.savefig("multiclass_notmnist_{}_loss_comparison.pdf".format(r), format="pdf")




plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
for r in learning_rates:
    v = np.load("2.2.1_{}_notmnist_v_acc.npy".format(r))
    t = np.load("2.2.1_{}_notmnist_t_acc.npy".format(r))
    plt.plot(v, linewidth=1.0, linestyle='-', label=("Learning Rate = {} Training Set Accuracy".format(r)))
    plt.plot(t, linewidth=1.0, linestyle='-', label=("Learning Rate = {} Validation Set Accuracy".format(r)))
plt.axhline(linewidth=2, color='r', y=1)
plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500", fontsize=10)
plt.xlabel("# of Epochs")
plt.ylabel("Training Loss (Cross-Entropy Loss)")
plt.legend(title="Training Rate", loc="lower right")
plt.grid('on', linestyle='-', linewidth=0.5)
plt.savefig("multiclass_notmnist_{}_acc_comparison.pdf".format(r), format="pdf")
