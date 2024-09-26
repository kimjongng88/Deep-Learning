import matplotlib.pylab as plt
import ast
import numpy as np


y = np.loadtxt("relu_acc.txt", dtype=float)
y2 = np.loadtxt("relu_loss.txt", dtype = float)
y3 = np.loadtxt("sig_loss.txt", dtype=float)


#plt.yticks(np.arange(0, 1, 0.1))
plt.plot(y,label = 'BS = 100',color = 'r')
plt.plot(y2,label = 'BS = 200',color = 'g')
plt.plot(y3,label = 'BS = 50',color = 'y')
plt.legend(fontsize=15)
plt.ylim(0,1)
#plt.yticks(np.arange(0.1, 1, 0.02))

plt.xlabel("EPOCH")
plt.ylabel("Accuracy")


plt.show()