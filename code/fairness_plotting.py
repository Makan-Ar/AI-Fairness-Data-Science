import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

# set to dataset name to be plotted
dataset_name = "credit-default"


result_file_to_read = '../results/{}/fair-measures-1.pckl'.format(dataset_name)
save_plot_path = '../results/plots'

with open(result_file_to_read, 'rb') as f:
    fairness_measures = pickle.load(f)

colors = {"DT": 'r', "RF": 'g', "GNB": 'y', "KNN": 'b', "LR": 'orange', "MLP": 'purple', "SVC": 'c', "GPC": 'm'}

y_lables = {'FPR': 'Mean False Positive Rate', 'FNR': 'Mean False Negative Rate',
            'DPVR': 'Mean Demographic Parity\nViolation Rate', 'EOVR': 'Mean Equality of Opportunity\nViolation Rate'}

names = ['FPR', 'FNR', 'DPVR', 'EOVR']
p_features = []

for measure in names:
    for classifier in fairness_measures[measure]:
        p_features = list(fairness_measures[measure][classifier].keys())
        break
    break

N = len(p_features)

for measure in names:
    num_algo = len(fairness_measures[measure])

    plt.rc('legend', fontsize=13)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    fig, ax = plt.subplots()
    ind = np.arange(N)
    width = 0.9 / num_algo

    for i, classifier in enumerate(fairness_measures[measure]):
        m = list(fairness_measures[measure][classifier].values())
        plt.bar(ind + i * width, m, width, label=classifier, color=colors[classifier])

    plt.xticks(ind - width / 2 + (width * num_algo) / 2, p_features)

    plt.ylabel(y_lables[measure], fontsize=20)

    plt.legend(ncol=math.floor(num_algo / 2), bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0)

    plt.tight_layout()
    current_fig = plt.gcf()
    current_fig.savefig("{}/{}-{}.pdf".format(save_plot_path, dataset_name, measure), bbox_inches="tight")
    plt.clf()
