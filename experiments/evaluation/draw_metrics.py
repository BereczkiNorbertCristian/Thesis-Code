import cfg
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm


def read_confusion_matrix(fname=cfg.RESULTS_PATH):
    ret = []
    with open(fname, 'r') as f:
        for line in f:
            ret.append([int(el) for el in line.strip().split(' ')])
    return ret

# what = 0 for precision
# what = 1 for recal


def compute_prec_rec(conf_arr, what=0):
    return sum(np.diag(conf_arr) / np.sum(conf_arr, axis=what)) / conf_arr.shape[0]


def compute_acc(conf_arr):
    return sum(np.diag(conf_arr)) / np.sum(conf_arr)

def lst_csv(lst, do_rounding = False):
	to_put = ""
	for el in lst:
		if do_rounding:
			el = round(el,3)
		to_put += str(el) + ","
	return to_put

def compute_stats(conf_arr, plot_table):
    avg_precision = compute_prec_rec(conf_arr, 0)
    avg_recall = compute_prec_rec(conf_arr, 1)
    acc = compute_acc(conf_arr)
    f1 = 2 * ((avg_precision * avg_recall) / (avg_precision + avg_recall))

    print("Stats:")
    print("Average precision: {0}".format(avg_precision))
    print("Average recall: {0}".format(avg_recall))
    print("Accuracy: {0}".format(acc))
    print("F1 score: {0}".format(f1))
    print("End...")

    precisions = np.diag(conf_arr) / np.sum(conf_arr, axis=0)
    recalls = np.diag(conf_arr) / np.sum(conf_arr, axis=1)
    f1scores = 2 * ((precisions*recalls) / (precisions + recalls))

    letters = cfg.labels_to_names.values()
    print("first line," + lst_csv(letters) + "\n")
    print("precision," + lst_csv(precisions, True) + "\n")
    print("recall," + lst_csv(recalls, True) + "\n")
    print("f1," + lst_csv(f1scores, True) + "\n")


plot_table = True
do_only_stats = True
conf_arr = read_confusion_matrix()
conf_arr = np.asarray(conf_arr)

compute_stats(conf_arr, plot_table)


if do_only_stats:
    exit(0)
alphabet = cfg.labels_to_names.values()

df_to_plot = pd.DataFrame(conf_arr, alphabet,
                          alphabet)
fig = plt.figure(figsize=(10, 7))
sns.heatmap(df_to_plot, annot=True)
# plt.savefig('resnet50_finetune.png')
plt.show()
