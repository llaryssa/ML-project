import numpy as np
import math
from sklearn import neighbors
from sklearn.metrics import r2_score

def readFile(path):
    file = open(path, 'r')
    data = []
    for line in file:
        if line[0] != '@':
            data.append(map(float, line.split(',')))
    data = np.array(data)
    return data[:,0:data.shape[1]-1], data[:,data.shape[1]-1]

def diskr(data, labels, theta, k_neighbors):
    raw_data = data
    raw_labels = labels
    knn = neighbors.KNeighborsRegressor(k_neighbors)
    print "starting... data.shape", data.shape

    ##### first part: remove outliers
    y_hat = knn.fit(data, labels).predict(data)
    PD = abs(y_hat - labels)
    outliers = PD > (1 - theta)*labels

    data = data[-outliers]
    labels = labels[-outliers]
    PD = PD[-outliers]

    print len(raw_data) - len(data), "outliers removed... data.shape", data.shape

    ###### second part: removing indistintive instances
    # sort in descending pd order
    sort_idx = PD.argsort()[::-1]
    data = data[sort_idx]
    labels = labels[sort_idx]


    # acho que da pra calcular isso tudo de uma vez sem ser iterativo
    i = 0
    while i < len(data):
        # T - {xi}
        data_without_xi = np.delete(data, i, axis=0)
        labels_without_xi = np.delete(labels, i)

        # print data.shape, data_without_xi.shape

        # suspeito que esse conjunto de treinamento (passado no fit) ta errado
        # tem que ver se sao os dados atuais removendo ou o dado inicial sem remocao
        # ta removendo gente demais
        y_hat = knn.fit(data, labels).predict(data_without_xi)
        y_hat_line = knn.fit(data_without_xi, labels_without_xi).predict(data_without_xi)

        Rbf = sum(map(lambda x: math.pow(x,2), (labels_without_xi - y_hat)))
        Raf = sum(map(lambda x: math.pow(x,2), (labels_without_xi - y_hat_line)))

        # print Raf, Rbf, (Raf - Rbf), "<=", theta, Raf, theta*Raf

        if (Raf - Rbf) <= theta * Raf:
            data = np.delete(data, i, axis=0)
            labels = np.delete(labels, i)
            i = i - 1
            # no algoritmo diz que agora precisa recalcular o centro do cara que a gente tirou
            # mas isso nao eh obvio pro knn? ele n faz isso sozinho?

        i = i + 1

    print "final: ", data.shape, labels.shape

    return data, labels




########################################################

dataset_path = "datasets/mortgage.dat"
# dataset_path = "datasets/abalone.dat"
# dataset_path = "datasets/ANACALT.dat"
data, labels = readFile(dataset_path)
print "data shape: ", data.shape, " | labels shape: ", labels.shape


### random split
p = np.random.permutation(len(data))
data = data[p]
labels = labels[p]

# defining each split for cross validation
cross_v = 10
gap = int(math.ceil(float(len(data))/cross_v))
cv_index = []
for i in range(0,cross_v):
    cv_index += [i]*gap
cv_index = np.array(cv_index[:len(data)])


r2_cross_validation = []


for cv in range(0,cross_v):
    train_idx = cv_index != cv
    test_idx = cv_index == cv

    train_data = data[train_idx,:]
    train_labels = labels[train_idx]

    test_data = data[test_idx,:]
    test_labels = labels[test_idx]

    print "training size: ", train_data.shape, "testing size: ", test_data.shape

    theta = 0.1
    k = 9

    data_, labels_ = diskr(train_data, train_labels, theta, k)

    knn = neighbors.KNeighborsRegressor(k)
    labels_hat = knn.fit(data_, labels_).predict(test_data)

    r2 = r2_score(test_labels, labels_hat)
    r2_cross_validation.append(r2)

    print cv, ":", r2
    print

print "r2: ", r2_cross_validation
print "mean: ", np.mean(r2_cross_validation)
