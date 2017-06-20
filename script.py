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
    # print "starting... data.shape", data.shape

    ##### first part: remove outliers
    y_hat = knn.fit(data, labels).predict(data)
    PD = abs(y_hat - labels)
    outliers = PD > (1 - theta)*labels

    data = data[-outliers]
    labels = labels[-outliers]
    PD = PD[-outliers]

    # print len(raw_data) - len(data), "outliers removed... data.shape", data.shape

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

    # print len(raw_data) - len(data), "instances removed"
    # print "final: ", data.shape, labels.shape
    # print "compression: ", float(len(data))/len(raw_data)

    return data, labels




########################################################
# dataset = "plastic"
# dataset = "mortgage"
# dataset = "concrete"
# dataset = "treasury"
# dataset = "ele-2"
# dataset = "wizmir"
dataset = "anacalt"

thetaByDataset = {"plastic":0.01,
                "mortgage":0.002,
                "concrete":0.001,
                "treasury":0.0005,
                "ele-2":0.007,
                "wizmir":0.001}

dataset_path = "datasets/" + dataset + ".dat"
data, labels = readFile(dataset_path)
print "testing dataset: ", dataset
print "data shape: ", data.shape, " | labels shape: ", labels.shape, "\n"


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
compression_cross_validation = []

# for cv in range(0,cross_v):
for cv in range(5,10,23):
    train_idx = cv_index != cv
    test_idx = cv_index == cv

    train_data = data[train_idx,:]
    train_labels = labels[train_idx]

    test_data = data[test_idx,:]
    test_labels = labels[test_idx]

    print "training size: ", train_data.shape, "testing size: ", test_data.shape


    for th in range(0,100,10):
        theta = float(th)/100

        # theta = thetaByDataset[dataset]
        k = 9

        try:
            data_, labels_ = diskr(train_data, train_labels, theta, k)
        except:
            print "erro theta", theta
            pass

        knn = neighbors.KNeighborsRegressor(k)
        labels_hat = knn.fit(data_, labels_).predict(test_data)

        r2 = r2_score(test_labels, labels_hat)
        r2_cross_validation.append(r2)

        compression = float(len(data_)) / len(train_data)
        compression_cross_validation.append(compression)

        print cv, " th:", theta, " | r2 =", r2, " c =", compression
        print

print "r2: ", r2_cross_validation
print "mean: ", np.mean(r2_cross_validation), "\n"

print "c: ", compression_cross_validation
print "mean: ", np.mean(compression_cross_validation)
