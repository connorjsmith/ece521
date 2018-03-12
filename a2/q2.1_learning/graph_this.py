import numpy as np
import matplotlib.pyplot as plt

learning_rates = [0.0001, 0.005, 0.001]
for r in learning_rates:
    plt.figure(figsize=(8,6), dpi=80)
    plt.subplot(1,1,1)
    v = np.load("log_q1_{}_valid_acc.npy".format(r))
    l = np.load("log_q1_{}_loss.npy".format(r))
    t = np.load("log_q1_{}_train_acc.npy".format(r))
    plt.plot(v, linewidth=1.0, linestyle='-', label="Validation Data Accuracy")
    plt.plot(t, linewidth=1.0, linestyle='-', label="Training Data Accuracy")
    plt.plot(l, linewidth=1.0, linestyle='-', label="Cross-Entropy Loss")
    plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500, Training Rate η = "+str(r), fontsize=10)
    plt.xlabel("# of Epochs")
    plt.ylabel("Training Loss (Cross-Entropy Loss)")
    plt.legend(title="Training Rate ()", loc="upper right")
    plt.grid('on', linestyle='-', linewidth=0.5)
    plt.savefig("learning_{}.pdf".format(r), format="pdf")
