from sklearn import svm


def svmClassifier(data):
    clf = svm.SVC()
    clf.fit(data[0].detach().numpy(), data[1].detach().numpy())
    return clf
