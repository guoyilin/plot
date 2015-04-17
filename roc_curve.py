import numpy as np
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='makes roc curve from binary-classify result')
    parser.add_argument('predict_file', help='file of predict and true label')
    parser.add_argument('jpg_file', help='file of save jpg')
    args = parser.parse_args()
    #read y_true,  y_score from file
    f = open(args.predict_file, 'r')
    lines = f.readlines()
    y_true = []
    y_predict = []
    for line in lines:
        y_true.append(int(line.split(" ")[-1].strip()))
        y_predict.append(float(line.split(" ")[1].strip()))
    f.close()

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_predict, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(args.jpg_file, dpi=100)