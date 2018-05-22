import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

# set to dataset name to be plotted
dataset_name = "credit-default"


result_file_to_read = '../results/{}/pred-accuracy-1.pckl'.format(dataset_name)
save_plot_path = '../results/plots'

with open(result_file_to_read, 'rb') as f:
    pred_accuracy = pickle.load(f)


plt.rc('legend', fontsize=13)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

fig, ax = plt.subplots()
ind = np.arange(len(pred_accuracy))
width = 0.5

m = list(pred_accuracy.values())
plt.bar(ind, m, width)

plt.ylim(ymin=math.floor(min(m) - 5))
plt.xticks(ind, list(pred_accuracy.keys()))
plt.ylabel("Prediction Accuracy", fontsize=20)

plt.tight_layout()
current_fig = plt.gcf()
current_fig.savefig("{}/{}-accuracy.pdf".format(save_plot_path, dataset_name))
plt.clf()
