from copyreg import pickle
import pickle as pk
from sklearn.decomposition import PCA
from skimage import exposure
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from skimage import exposure
from sklearn.cluster import KMeans

image = []

model_path = "rdf_nestimators_150_state_42.pkl"
img_path = "test_preson.png"

def show(image):
    if (image.shape[0] != 64):
        plt.imshow(image.reshape(64, 64), cmap="gray")
    else:
        plt.imshow(image, cmap="gray")
    plt.show()


class Predictor():
    model_path = "rdf_nestimators_150_state_42.pkl"
    _model = None
    _X_train = None
    _Y_train = None
    _X_valid = None
    _Y_valid = None
    _X_test = None
    _Y_test = None
    _pca = None
    one = None
    two = None

    def _load_dataset(self):
        olivetti = fetch_olivetti_faces()

        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
        train_valid_idx, test_idx = next(strat_split.split(olivetti.data, olivetti.target))
        X_train_valid = olivetti.data[train_valid_idx]
        y_train_valid = olivetti.target[train_valid_idx]
        X_test = olivetti.data[test_idx]
        y_test = olivetti.target[test_idx]

        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
        train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))
        X_train = X_train_valid[train_idx]
        y_train = y_train_valid[train_idx]
        X_valid = X_train_valid[valid_idx]
        y_valid = y_train_valid[valid_idx]

        self._X_train = X_train
        self._Y_train = y_train
        self._X_valid = X_valid
        self._Y_valid = y_valid
        self._X_test = X_test
        self._Y_test = y_test

        return olivetti

    def _create_pca(self):
        pca = PCA(0.99)
        pca.fit(self._X_train)
        self._pca = pca

    def load(self, path):
        return (cv2.imread(path, 0))

    def save(self, image, path):
        plt.imsave(path, image, cmap="gray")
        return (path)

    def __init__(self) -> None:
        f = open(model_path , 'rb')
        self._model = pk.load(f)
        self._load_dataset()
        self._create_pca()

    def get_compressed_image(self, image):
        X_train_pca = self._pca.transform(image.reshape(1, 4096))
        return self._pca.inverse_transform(X_train_pca).reshape(64, 64)

    def get_equalized_image(self, image):
        img = exposure.equalize_adapthist(image, clip_limit=0.01).reshape(-1, 64)
        return img

    def predict(self, image):
        image = image / 255
        tmp = self._pca.transform(image)
        return self._model.predict(tmp)[0]

    def evaluate(self):
        x = self._pca.transform(self._X_test)
        y = self._Y_test

        current_pca = self._pca.transform(x)
        preds = self._model.predict(current_pca)

    def get_image_by_index(self, value):
        index = list(self._Y_test).index(value)
        return self._X_test[index].reshape(-1, 64)


# toto = Predictor()
# image = cv2.imread(img_path, 0)
# eq = toto.get_equalized_image(image)
# comp = toto.get_compressed_image(eq)
# lab = toto.predict(image.reshape(-1,4096))
# pred = toto.get_image_by_index(lab)
# toto.evaluate()
# import pdb; pdb.set_trace()