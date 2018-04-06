# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:35:48 2018

@author: Jeffrey
"""

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6), dpi=80)

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85, left = .15, right = .85, bottom = .15)
ax.set_title('2 Hidden Layers of 500 Nodes')

ax.set_xlabel('# of epochs')
ax.set_ylabel('validation classification error')

data = np.load("Q1-2-2.npy")

ax.plot(1 - np.transpose(data[0,:,:,2]))
etas = [0.01, 0.005, 0.001, 0.0005, 0.0001];
ax.legend(["$\eta$ = "+str(i) for i in etas], loc='upper right')


ax.axis([0, 166, 0, 1])

plt.show()

fig.savefig("2_hidden_layers_val_cacc.pdf")