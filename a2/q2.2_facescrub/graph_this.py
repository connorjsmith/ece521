import numpy as np
import matplotlib.pyplot as plt

learning_rates = [0.01, 0.005, 0.001, 0.0001]
decay_rates = [0.1, 0.01, 0.001, 0]
plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
for r in learning_rates:
    for d in decay_rates:
        l = np.load("2.2.2_r{}_d{}_facescrub_loss.npy".format(r, d))
        plt.plot(l, linewidth=1.0, linestyle='-', label=("Learning Rate = {} Decay = {} Loss".format(r, d)))
plt.title("Mini-Batch Size B=300", fontsize=10)
plt.xlabel("# of Epochs")
plt.ylabel("Training Loss (Softmax Loss)")
plt.legend(title="Training Rate and Decay", loc="lower right")
plt.grid('on', linestyle='-', linewidth=0.5)
plt.savefig("multiclass_facescrub_loss_comparison.pdf", format="pdf")


'''
plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
for r in learning_rates:
    v = np.load("2.2.1_{}_notmnist_v_acc.npy".format(r))
    plt.plot(v, linewidth=1.0, linestyle='-', label=("Learning Rate = {} Validation Set Accuracy".format(r)))
plt.axhline(linewidth=2, color='r', y=1)
plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500", fontsize=10)
plt.xlabel("# of Epochs")
plt.ylabel("Accuracy (Percentage Correct)")
plt.legend(title="Training Rate", loc="lower right")
plt.grid('on', linestyle='-', linewidth=0.5)
plt.savefig("multiclass_notmnist_v_acc_comparison.pdf", format="pdf")


plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
for r in learning_rates:
    t = np.load("2.2.1_{}_notmnist_test_acc.npy".format(r))
    plt.plot(t, linewidth=1.0, linestyle='-', label=("Learning Rate = {} Test Set Accuracy".format(r)))
plt.axhline(linewidth=2, color='r', y=1)
plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500", fontsize=10)
plt.xlabel("# of Epochs")
plt.ylabel("Accuracy (Percentage Correct)")
plt.legend(title="Training Rate", loc="lower right")
plt.grid('on', linestyle='-', linewidth=0.5)
plt.savefig("multiclass_notmnist_test_acc_comparison.pdf", format="pdf")



plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
for r in learning_rates:
    t = np.load("2.2.1_{}_notmnist_train_acc.npy".format(r))
    plt.plot(t, linewidth=1.0, linestyle='-', label=("Learning Rate = {} Training Set Accuracy".format(r)))
plt.axhline(linewidth=2, color='r', y=1)
plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500", fontsize=10)
plt.xlabel("# of Epochs")
plt.ylabel("Accuracy (Percentage Correct)")
plt.legend(title="Training Rate", loc="lower right")
plt.grid('on', linestyle='-', linewidth=0.5)
plt.savefig("multiclass_notmnist_train_acc_comparison.pdf", format="pdf")


plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
t_l = np.load("2.2.1_0.005_notmnist_t_loss.npy".format(r))
v_l = np.load("2.2.1_0.005_notmnist_v_loss.npy".format(r))
plt.plot(t_l, linewidth=1.0, linestyle='-', label=("Training Set Loss"))
plt.plot(v_l, linewidth=1.0, linestyle='-', label=("Validation Set Loss"))
plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500", fontsize=10)
plt.xlabel("# of Epochs")
plt.ylabel("Softmax Loss")
plt.legend(title="Dataset", loc="upper right")
plt.grid('on', linestyle='-', linewidth=0.5)
plt.savefig("multiclass_notmnist_chosen_loss_comparison.pdf", format="pdf")

'''

