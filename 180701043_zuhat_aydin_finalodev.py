import numpy as np
from sklearn.impute import SimpleImputer
from math import ceil
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score, r2_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time


def veri_yukle(veri_dosyasi, ayrac=','):
    def degis(x):
        if x.decode("utf-8") == "EVET":
            return 1
        else:
            return 0

    veri = np.genfromtxt(veri_dosyasi, delimiter=ayrac, dtype=float, skip_header=True, converters={-1: degis})

    # boş değerleri doldurma
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp.fit(veri)
    veri = imp.transform(veri)

    train_sayisi = ceil(len(veri) * 0.7)

    # from sklearn.feature_selection import RFE
    # from sklearn.svm import SVR
    # estimator = SVR(kernel="linear")
    # selector = RFE(estimator, n_features_to_select=13, step=1)
    # selector = selector.fit(X, Y)
    # print(X)
    # print(selector.support_)

    # veri = np.delete(veri, [8,12], axis=1)
    X = veri[:, :-1]
    Y = veri[:, -1]

    # değerleri aralığa sığdırma
    # X = StandardScaler().fit_transform(X)

    # embedding = MDS(n_components=1)
    # X = embedding.fit_transform(X)
    xtrain = X[:train_sayisi, :]
    ytrain = Y[:train_sayisi]
    xtest = X[train_sayisi:, :]
    ytest = Y[train_sayisi:]
    return X, Y, xtrain, ytrain, xtest, ytest


def verigorsellestir(x, y):
    # değişken sayısını indirgeme
    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(x)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Yeni değişken 1', fontsize=15)
    ax.set_ylabel('Yeni değişken 2', fontsize=15)
    ax.set_title('MDS 2 DEĞİŞKENLİ SONUÇ', fontsize=20)

    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, s=100)

    plt.show()


def lineerReg(X, y, tekrar=500, alfa=0.0001):

    veri_sayisi = len(X)
    bias = 0
    coefs = np.zeros(len(X[0]))
    for i in range(tekrar):
        h = np.dot(X, coefs) + bias
        # print(1/veri_sayisi*sum([abs(val) for val in (y-h)]))
        bias = bias - alfa * ((1 / veri_sayisi) * sum(h - y))
        coefs = coefs - alfa * ((1 / veri_sayisi) * np.dot((h - y), X))

    return coefs, bias


def lineer_tahmin(X, coefs, bias):
    tahmini = []
    for i in range(len(X)):
        s = np.dot(X[i], np.transpose(coefs)) + bias
        tahmini.append(s.round())

    return tahmini


def lineer(xtrain, ytrain, xtest, ytest):
    basla = time.time()
    coefs, bias = lineerReg(xtrain, ytrain)
    bitir = time.time()
    print("harcanan süre: ", bitir-basla)
    print(coefs, bias)
    test_lineer_tahmini = lineer_tahmin(xtest, coefs, bias)
    test_sonuc = r2_score(ytest, test_lineer_tahmini)
    print("test doğruluk :", test_sonuc)
    print("lineer için confusion matrix (test)", confusion_matrix(ytest, test_lineer_tahmini))

    egitim_lineer_tahmini = lineer_tahmin(xtrain, coefs, bias)
    egiitim_sonuc = r2_score(ytrain, egitim_lineer_tahmini)
    print("eğitim doğruluk :", egiitim_sonuc)
    print("lineer için confusion matrix (eğitim)", confusion_matrix(ytrain, egitim_lineer_tahmini))


def sigmoid_fonksiyon(z):
    f = 1/(1 + np.exp(-z))
    return f


def lojistikreg(X, y, tekrar=1000, alfa=0.2):
    veri_sayisi = len(X)
    bias = 0
    coefs = np.zeros(len(X[0]))
    for i in range(tekrar):
        h = sigmoid_fonksiyon(np.dot(X, coefs) + bias)
        # print(1/veri_sayisi*sum([abs(val) for val in (y-h)]))
        bias = bias - alfa * ((1 / veri_sayisi) * sum(h - y))
        coefs = coefs - alfa * ((1 / veri_sayisi) * np.dot((h - y), X))
    return coefs, bias


def lojistik_tahmin(X, t0, teta):
    tahmin_listesi = []
    test_sayisi = len(X)
    for i in range(test_sayisi):
        s = np.dot(X[i], teta) + t0
        tahmini = sigmoid_fonksiyon(s)
        if tahmini <= 0.5:
            tahmin_listesi.append(0)
        else:
            tahmin_listesi.append(1)

    return tahmin_listesi


def lojistik(xtrain, ytrain, xtest, ytest):
    basla = time.time()
    coefs, bias = lojistikreg(xtrain, ytrain)
    bitir = time.time()
    print("harcanan süre: ", bitir - basla)
    print(coefs, bias)

    test_tahmin = lojistik_tahmin(xtest, bias, coefs)
    print("lojistik için confusion matrix (test)", confusion_matrix(ytest, test_tahmin))
    print("eğitim doğruluk :", accuracy_score(ytest, test_tahmin))

    egitim_tahmin = lojistik_tahmin(xtrain, bias, coefs)
    print("lojistik için confusion matrix (eğitim)", confusion_matrix(ytrain, egitim_tahmin))
    print("test doğruluk :", accuracy_score(ytrain, egitim_tahmin))


def svm(X, y, xtest, ytest):
    basla = time.time()
    clf = SVC(C=2, kernel='rbf')
    clf.fit(X, y)
    bitir = time.time()
    print("harcanan süre: ", bitir - basla)
    print("Test Skoru: ", clf.score(xtest, ytest))
    print("Eğitim Skoru: ", clf.score(X, y))

    test_tahmini = clf.predict(xtest)
    print("SVM için confusion matrix (test)", confusion_matrix(ytest, test_tahmini))

    egitim_tahmini = clf.predict(X)
    print("SVM için confusion matrix (eğitim)", confusion_matrix(y, egitim_tahmini))


def ann(X, y, xtest, ytest):
    basla = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation="tanh", solver="adam", random_state=1, max_iter=1000)
    clf.fit(X, y)
    bitir = time.time()
    print("harcanan süre: ", bitir - basla)
    print("Test Skoru: ", clf.score(xtest, ytest))
    print("Eğitim Skoru: ", clf.score(X, y))
    print(confusion_matrix(ytest, clf.predict(xtest)))

    egitim_tahmini = clf.predict(X)
    print("yapay sinir ağı için confusion matrix (eğitim)", confusion_matrix(y, egitim_tahmini))
    test_tahmini = clf.predict(xtest)
    print("yapay sinir ağı için için confusion matrix (test)", confusion_matrix(ytest, test_tahmini))

    print(clf.coefs_)
    print(clf.intercepts_)


def knn(X, y, xtest, ytest):
    kmeans = KMeans(n_clusters=3, random_state=1)
    kmeans.fit(X)
    hayir = 0
    evet = 0
    karmasa = 0
    a = kmeans.predict(xtest)
    for i in range(len(ytest)):
        if ytest[i] == 0 and a[i] == 1:
            hayir += 1
        elif ytest[i] == 1 and a[i] == 2:
            evet += 1
        elif ytest[i] == 0 and a[i] == 2:
            karmasa += 1
        elif ytest[i] == 1 and a[i] == 1:
            karmasa += 1

    print(evet, hayir, karmasa, len(xtest))
    dogruluk = (evet + hayir) / (evet+hayir+karmasa)
    print(dogruluk)
    print(kmeans.labels_)
    print(y)


def main():
    X, Y, xtrain, ytrain, xtest, ytest = veri_yukle("NBA5Sene.csv")

    # verigorsellestir(X, Y)

    # lineer(xtrain, ytrain, xtest, ytest)

    # lojistik(xtrain, ytrain, xtest, ytest)

    # svm(xtrain, ytrain, xtest, ytest)

    # ann(xtrain, ytrain, xtest, ytest)

    # knn(xtrain, ytrain, xtest, ytest)


if __name__ == '__main__':
    main()
