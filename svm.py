import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class SuppVectMach(object):
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
        self.model = SVC(probability=True)
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

    svc = SuppVectMach()
    svc.load_data(data_fp)
    svc.train()

    roc_svc = []
    threshold = 0

    for i in range(0, 5):
        svc.eval(threshold + i * 0.005)

        tn = svc.confusion['tn']
        fp = svc.confusion['fp']
        fn = svc.confusion['fn']
        tp = svc.confusion['tp']

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        roc_svc.append([threshold, tn, fp, fn, tp, fpr, tpr, precision, recall])

    roc_svc_df = pd.DataFrame(roc_svc, columns=['Threshold', 'tn', 'fp', 'fn', 'tp','fpr', 'tpr', 'precision', 'recall'])
    print(roc_svc_df)
    roc_svc_df.to_csv('./output/roc_svc_df.csv', index=False)


if __name__ == '__main__':
    main()
