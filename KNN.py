import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from fastdtw import fastdtw
import util


class KNN:
    def __init__(self, K):
        self.trajectories = []           # list of all known trajectories
        self.trajectory_categories = []  # parallel list of their categories
        self.categories = set()          # set of all known trajectories
        self.K = K

    def fit(self, train_set, train_categories):
        self.categories = set(train_categories)
        for i in range(len(train_set)):
			self.trajectories.append(train_set[i])
			self.trajectory_categories.append(train_categories[i])

    def predict(self, test_set):
        prediction = []
        for newpoint in test_set:
            prediction.append(self.predict_for_one(newpoint))
        return prediction

    def predict_for_one(self, unknown_trajectory):
        if unknown_trajectory is None:
            return None
        min5 = [(float('inf'), None)] * self.K     # top-K minimum-distanced neighbours' categories
        for i in range(len(self.trajectories)):
            distance, _ = fastdtw(self.trajectories[i], unknown_trajectory, dist=util.harversineDist)
            maxdist, maxpos = max((v[0], j) for j, v in enumerate(min5))
            if distance < maxdist:                 # if the i-th trajectory is better than the furthest neighbor so far then replace the latter
                min5[maxpos] = (distance, self.trajectory_categories[i])
        category_count = {c: 0 for c in self.categories}
        for i in range(self.K):                 # Voting scheme: majority voting
            if min5[i][1] is not None:
            	category_count[min5[i][1]] += 1
        maxvotes = -1
        maxcat = None
        for cat in self.categories:
            if category_count[cat] > maxvotes:
                maxvotes = category_count[cat]
                maxcat = cat
        return maxcat


def crossvalidation(X, y, K):
    skf = StratifiedKFold(n_splits=10)
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = KNN(K)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        curr_acc = accuracy_score(y_test, y_pred)
        scores.append(curr_acc)
        print "Fold accuracy: " + str(curr_acc)
    return np.mean(scores)
