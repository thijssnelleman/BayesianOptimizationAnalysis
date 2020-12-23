import matplotlib.pyplot as plt
import numpy as np
import statistics

save_path = "/content/drive/MyDrive/LIACS/AML/"
cnnrandom = "Random_CNN-Result-RUN-"
cnnBO = "CNN-Result-RUN-"
mlprandom = "RandomMLP-Result-RUN-"
mlpBO = "MLP-Result-RUN-"
x = np.array(range(50))
def loadData(loc, filename):
    y = np.zeros(50)
    std_lists = [[] for x in range(50)]

    for res in range(10):
        fl = loc + filename + str(res+1) + ".npy"
        file = np.load(fl)
        for index, val in enumerate(file):
            std_lists[index].append(val)
            y[index] += val
    y /= 10 #take the average
    std = np.array([statistics.stdev(std_lists[val]) for val in range(50)])

    best = 0
    for n in range(1, len(x)):
        if y[n] <= y[best]:
            best = n
        else:
            y[n] = y[best]
            std[n] = std[best]
    return y, std

BOy, BOstd = loadData(save_path, cnnBO)
Ry, Rstd = loadData(save_path, cnnrandom)

plt.errorbar(x, BOy, BOstd, linestyle='solid', marker='^', color='r', label='Bayesion Optimization')
plt.errorbar(x, Ry, Rstd, linestyle='dashed', marker='^', color='b', label='Random Search')
data1 = {
    'x': list(range(0, len(Ry))),
    'y1': [Y - e for Y, e, in zip(Ry,Rstd)],
    'y2': [Y + e for Y, e, in zip(Ry,Rstd)]
}
plt.fill_between(**data1, alpha=0.25)
data2 = {
    'x': list(range(0, len(Ry))),
    'y1': [Y - e for Y, e, in zip(BOy,BOstd)],
    'y2': [Y + e for Y, e, in zip(BOy,BOstd)]
}
plt.fill_between(**data2, alpha=0.25)
plt.xlabel("Epochs")
plt.ylabel("Loss value")
plt.legend()

plt.show()