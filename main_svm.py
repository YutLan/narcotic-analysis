from sklearn import svm
from sklearn import preprocessing
from tqdm import tqdm

from utils import nacro_train_test_split

def svm_classification(train_data, test_data, train_label, test_label, c):
    res = {}
    res['c'] = 0
    res['acc'] = 0
    res['p_label'] = 0
    res['test_label'] = test_label 
    clf = svm.LinearSVC(C=c, max_iter=100000,)
    # clf = svm.LinearSVC(C=c, dual=False)
    clf.fit(train_data, train_label)
    p_labels = clf.predict(test_data)
    score = clf.score(test_data, test_label)
    decision_value = clf.decision_function(test_data)
    if score > res['acc']:
        res['acc'] = score
        res['c'] = c
        res['p_label'] = p_labels
        res['decision_val'] = decision_value 
    return res  

def main():
    train_data, test_data, train_label, test_label = nacro_train_test_split()
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data = min_max_scaler.fit_transform(train_data)
    test_data = min_max_scaler.transform(test_data)
    res_all = {}
    C = [2**c for c in range(-10, 10)]
    print("parameters C list is: {}".format(C))
    for c in tqdm(C):
        res = svm_classification(train_data, test_data, train_label, test_label, c)
        res_all[c] = res['acc']
    print(max(res_all.values()))



if __name__ == '__main__':
    main()
