import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from heapq import heappush, heappop
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
        for i in range(0, len(train_set)):
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
        heap = []                        # minheap
        for i in range(0, len(self.trajectories)):
            heappush(heap, (fastdtw(self.trajectories[i], unknown_trajectory, dist=util.harversineDist), self.trajectory_categories[i]))
        category_count = {c: 0 for c in self.categories}
        for _ in range(0, self.K):       # only pop top-K (smallest K distances)
            _, c = heappop(heap)
            category_count[c] += 1
        max = -1
        maxcat = None
        for cat in self.categories:      # Voting scheme: majority voting
            if category_count[cat] > max:
                max = category_count[cat]
                maxcat = cat
        return maxcat


def crossvalidation(X, y, K):
	skf = StratifiedKFold(n_splits=10)
	scores = []
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf = KNN.KNN(K)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		scores.append(accuracy_score(y_test, y_pred))
	return np.mean(scores)
