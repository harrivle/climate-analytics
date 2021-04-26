import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class LogReg(object):
    def __init__(self):
        self.data = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.confusion = None
        self.y_predict_df = None

    def load_data(self, fp):
        self.data = pd.read_csv(fp)

        x = self.data['ave_temp']
        y = self.data['disaster_occurrence']

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, random_state=1)

    def train(self):
        self.model = LogisticRegression()
        self.model.fit(self.x_train.to_numpy().reshape(-1, 1), self.y_train.to_numpy().reshape(-1, 1))

    def eval(self, threshold=0.5):
        y_predict_prob = self.model.predict_proba(self.x_test.to_numpy().reshape(-1, 1))
        y_predict_prob = np.round(y_predict_prob[:, 1:], 5)
        y_predict = np.where(y_predict_prob >= threshold, 1, 0)

        self.y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'])

        conf = confusion_matrix(self.y_test, y_predict).ravel()
        self.confusion = {'tn': conf[0], 'fp': conf[1], 'fn': conf[2], 'tp': conf[3]}


def main():
    data_fp = './data/test_disasters_temp_state_month.csv'

    lr = LogReg()
    lr.load_data(data_fp)
    lr.train()

    roc_lr = []
    threshold = 0

    for i in range(0, 10):
        lr.eval(threshold + i * 0.005)

        tn = lr.confusion['tn']
        fp = lr.confusion['fp']
        fn = lr.confusion['fn']
        tp = lr.confusion['tp']

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        roc_lr.append([threshold, tn, fp, fn, tp, fpr, tpr, precision, recall])

    roc_lr_df = pd.DataFrame(roc_lr, columns=['Threshold', 'tn', 'fp', 'fn', 'tp','fpr', 'tpr', 'precision', 'recall'])
    print(roc_lr_df)
    roc_lr_df.to_csv('./output/roc_lr_df.csv', index=False)


if __name__ == '__main__':
    main()
