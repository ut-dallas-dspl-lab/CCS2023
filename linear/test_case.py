import numpy as np
import os
import pickle

# classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

# data loader
from data_util import MNISTMultiLabelSample 
from attack import Attack

class TestCase():

    def __init__(self,data_name, X_train, y_train, X_test, y_test, C=3):
       self.data_name = data_name
       self.problems = ['even_odd', 'greater', 'zero']
       #self.sgd_clf = [SGDClassifier() for i in range(C)]
       self.sgd_clf = [CalibratedClassifierCV() for i in range(C)]
       self.X_train = X_train
       self.y_train = y_train
       self.X_test = X_test
       self.y_test = y_test 
       self.y_test_all = []


    def train_models(self):
       if self.data_name == 'mnist':
           prefix = 'mnist_model/'
           #problems = ['even_odd', 'greater', 'zero']

           for c in self.problems:
               new_sample = MNISTMultiLabelSample(c)
               X_train, y_train, X_test, y_test = new_sample.make_mnist_samples(self.X_train, self.y_train, self.X_test, self.y_test)
               self.y_test_all.append(y_test)
               filename = prefix+'model_'+ c +'.sav'
               if os.path.isfile(filename):
                   print('[INFO:] loading' + filename)
                   self.sgd_clf[self.problems.index(c)]=pickle.load(open(filename,'rb'))
               else:
                   print('[INFO:] training' + filename)
                   clf = SGDClassifier(random_state=42)
                   self.sgd_clf[self.problems.index(c)] = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
                   self.sgd_clf[self.problems.index(c)].fit(X_train, y_train)
                   pickle.dump(self.sgd_clf[self.problems.index(c)], open(filename, 'wb'))
               print(self.sgd_clf[self.problems.index(c)].score(X_test, y_test))


    def attack_models(self, attack_type, n_selected, n_attk_samples):
       if self.data_name == 'mnist':
           #problems = ['even_odd', 'greater', 'zero']
           attack_case  = Attack(attack_type);
           lb = 0
           ub = 1
           start_x, selected, adv_x = attack_case.attack(self.X_test, self.y_test_all, self.sgd_clf, self.problems, n_selected, n_attk_samples, lb,ub)
       return start_x, selected, adv_x



    def get_metrics(self, n_start, n_samples, adv_x, C):

        original_acc=[]
        attacked_acc=[]
        original_recall=[]
        attacked_recall=[]

        n_features = self.X_train.shape[1]*self.X_train.shape[2]

        for c in range(0,C):

            # data for problem c
            X_test_c = np.reshape(self.X_test, (-1, n_features))
            y_test_c = self.y_test_all[c]

            # accuracy and recall on the original
            orig_y_pred = self.sgd_clf[c].predict(X_test_c[n_start:n_start+n_samples])
            tn,fp,fn,tp = confusion_matrix(y_test_c[n_start:n_start+n_samples], orig_y_pred, labels=[-1,1]).ravel()
            orig_recall = tp/(tp+fn)
            orig_acc = self.sgd_clf[c].score(X_test_c[n_start:n_start+n_samples], y_test_c[n_start:n_start+n_samples])
            original_recall.append(orig_recall)
            original_acc.append(orig_acc)

            # recall and accuracy on the attacked
            attk_y_pred = self.sgd_clf[c].predict(adv_x)
            tn,fp,fn,tp = confusion_matrix(y_test_c[n_start:n_start+n_samples], attk_y_pred, labels=[-1,1]).ravel()
            attk_recall = tp/(tp+fn)
            attk_acc = self.sgd_clf[c].score(adv_x, y_test_c[n_start:n_start+n_samples])
            attacked_recall.append(attk_recall)
            attacked_acc.append(attk_acc)

            print('Target Concept: ', c)
            #print('total: ', np.count_nonzero(y_test[n_start:n_start+n_samples]==c))
            print(self.sgd_clf[c].score(X_test_c[n_start:n_start+n_samples], y_test_c[n_start:n_start+n_samples]))
            print(self.sgd_clf[c].score(adv_x, y_test_c[n_start:n_start+n_samples]))

        return original_acc, attacked_acc, original_recall, attacked_recall


