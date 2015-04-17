import numpy as np
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='makes precision-recall curve from binary-classify result')
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
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_predict, pos_label=1)
    average_precision = metrics.average_precision_score(y_true, y_predict)

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(recall, precision, label='Precision-recall curve (area = %0.2f)' % average_precision)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(loc="lower left")
    #plt.show()
    plt.savefig(args.jpg_file, dpi=100)
