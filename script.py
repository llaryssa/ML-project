import numpy as np
import math
from sklearn import neighbors

def readFile(path):
    file = open(path, 'r')
    data = []
    for line in file:
        if line[0] != '@':
            data.append(map(float, line.split(',')))
    data = np.array(data)
    return data[:,0:data.shape[1]-1], data[:,data.shape[1]-1]

def diskr(data, labels, theta, k_neighbors):
    knn = neighbors.KNeighborsRegressor(k_neighbors)
    print "starting... data.shape", data.shape

    ##### first part: remove outliers
    y_hat = knn.fit(data, labels).predict(data)
    PD = abs(y_hat - labels)
    outliers = PD > (1 - theta)*labels

    data = data[-outliers]
    labels = labels[-outliers]
    PD = PD[-outliers]

    print "outliers removed... data.shape", data.shape

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

        print data.shape, data_without_xi.shape

        # suspeito que esse conjunto de treinamento (passado no fit) ta errado
        # tem que ver se sao os dados atuais removendo ou o dado inicial sem remocao
        # ta removendo gente demais
        y_hat = knn.fit(data, labels).predict(data_without_xi)
        y_hat_line = knn.fit(data_without_xi, labels_without_xi).predict(data_without_xi)

        Rbf = sum(map(lambda x: math.pow(x,2), (labels_without_xi - y_hat)))
        Raf = sum(map(lambda x: math.pow(x,2), (labels_without_xi - y_hat_line)))

        if (Raf - Rbf) <= theta * Raf:
            # tem que ver como fica o for(i) depois disso
            data = np.delete(data, i, axis=0)
            labels = np.delete(labels, i)
            i = i - 1
            # no algoritmo diz que agora precisa recalcular o centro do cara que a gente tirou
            # mas isso nao eh obvio pro knn? ele n faz isso sozinho?

        i = i + 1

    print "final: ", data.shape




########################################################

dataset_path = "datasets/abalone.dat"
data, labels = readFile(dataset_path)
print "data shape: ", data.shape, " | labels shape: ", labels.shape


### random split
p = np.random.permutation(len(data))
data = data[p]
labels = labels[p]

split = len(data) * 0.2
train_data = data[:int(split),:]
train_labels = labels[:int(split)]

test_data = data[int(split):,:]
test_labels = labels[int(split):]

theta = 0.1
k = 9
diskr(train_data, train_labels, theta, k)
