
import numpy as np

class KNN(object):

    def __init__(self, k):
        # initialize the number of nerghbors
        self.k = k

    def get_distance(self, point1, point2):
        # assume feature_v1 and feature_v2 are both numpy.array
        # TODO: try to use built-in numpy functions to get the distance between two points
        distance = np.linalg.norm(point1 - point2)
        return distance

    def get_all_distance(self, train_data, test_point):
        train_points = train_data[:, :-1].astype(np.float32)
        # Now you are given the whole train_points as a matrix
        # (np.ndarray in shape N, d)
        # N is the sample size, d is the feature size
        # test_point is the feature for test_point, in shape (d,)
        # You want to get the distance between your test point to all the training point at the same time
        # TODO: use broadcasting to get the difference between all the rows of train_points and test_point
        differences = train_points - test_point
        # TODO: the differences should also be in shape (N, d), use the built-in norm function on the
        # final axis to get the distances, result in shape (N,)
        distances = np.linalg.norm(differences, axis=-1)
        return distances

    def find_k_neighbors(self, train_data, test_data):
        all_distances = self.get_all_distance(train_data, test_data)
        result = np.stack([all_distances, train_data[:, -1]], -1)
        # sort the result by the distance
        result = result[np.argsort(result[:, 0]), :]
        # return the top k prediction
        topk = result[:self.k, :]
        return topk

    def get_prediction(self, topk):
        categories, count = np.unique(topk[:, -1], return_counts=True)
        return categories[np.argmax(count)]

    def fit_and_predict(self, train_data, test_data, topk=None):
        if topk is None:
            topk = self.find_k_neighbors(train_data, test_data)
        prediction = self.get_prediction(topk)
        return prediction


train_dataset = np.array([
[2, 2, 2, 2, 'a'],
[4, 4, 4, 4, 'b'],
[3, 3, 3, 3, 'b'],
[1, 1, 1, 1, 'a']
])
test_datapoint = np.array([5, 5, 5, 5])
knn_finder = KNN(k = 3)
prediction = knn_finder.fit_and_predict(train_dataset, test_datapoint)
print("final prediction of the test datapoint is:", prediction)
