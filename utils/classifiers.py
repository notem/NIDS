from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


def _process_results(truths, preds):
    m = {}
    for truth, pred in list(zip(truths, preds)):
        d = m.get(truth, {})
        d[pred] = d.get(pred, 0) + 1
        m[truth] = d
    return m


class Classifier(object):
    """Common interface for all IDS classifiers """
    trained = False

    def train(self, X_train, Y_train):
        pass

    def evaluate(self, X_test, Y_test):
        pass


class RandomForest(Classifier):
    """RandomForest (supervised) classifier for use in misuse-based NIDS classification"""
    num_trees = 50
    model = None
    le = None
    cm = None

    def train(self, X_train, Y_train):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(Y_train)
        Y_train = self.le.transform(Y_train)

        self.model = RandomForestClassifier(n_jobs=2,
                                            n_estimators=self.num_trees,
                                            oob_score=True,
                                            verbose=True)
        self.model.fit(X_train, Y_train)
        self.trained = True
        print("oob_score = ", self.model.oob_score_)

    def evaluate(self, X_test, Y_test):
        # make predictions
        results = self.le.inverse_transform(self.model.predict(X_test))
        # create confusion matrix
        self.cm = confusion_matrix(Y_test, results)
        # process results into a dict
        processed = _process_results(Y_test, results)
        return processed


class SVM(Classifier):
    """One-Class SVM (unsupervised) classifier for use in anomaly-based NIDS classification"""
    nu = 0.02
    gamma = 'auto'
    kernel = 'rbf'
    max_iter = -1
    model = None
    cm = None

    def train(self, X_train, Y_train):
        # train a one-class model
        self.model = OneClassSVM(nu=self.nu,
                                 gamma=self.gamma,
                                 max_iter=self.max_iter,
                                 kernel=self.kernel,
                                 verbose=True)
        self.model.fit(X_train)
        self.trained = True

    def evaluate(self, X_test, Y_test):
        # make predictions
        results = ['Normal' if r > 0 else 'Attack' for r in self.model.predict(X_test)]
        # create confusion matrix
        self.cm = confusion_matrix(Y_test, results)
        # process results into a dict
        return _process_results(Y_test, results)
