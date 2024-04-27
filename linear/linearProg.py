import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# data, test, and plot modules
from data_util import MNISTMultiLabelSample
from test_case import TestCase
from plot_bar import plot_acc
from explainer import Explainer

# construct argument parser
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", default='mnist', choices=['mnist', 'fashion_mnist', 'faces'], help="dataset: mnist, fashion_mnist, face")
args = vars(ap.parse_args())


# load the original data
DATASET = args["data"]
if DATASET == 'mnist':
    data = tf.keras.datasets.mnist
elif DATASET == 'fashion_mnist':
    data = tf.keras.datasets.fashion_mnist
elif DATASET == 'face':
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

if DATASET == 'mnist' or DATASET == 'fashion_mnist':
    (X_train, y_train), (X_test, y_test) = data.load_data()
    n_features = X_train.shape[1]*X_train.shape[2]
    X_train = X_train/255.
    X_test = X_test/255.
elif DATASET == 'face':
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data.reshape(-1,h,w)
    resize_w = 28
    resize_h = 28
    resized_dimentions = (resize_w, resize_h)
    X_resize = np.ndarray(shape=(n_samples, resize_w, resize_h), dtype=int)
    for i in range(n_samples):
        X_resize[i] = cv2.resize(X[i], resized_dimentions, interpolation=cv2.INTER_AREA)
    n_features = resize_w*resize_h

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_classes: %d" % n_classes)
    print('target_names: ', target_names)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X_resize, y, test_size=0.25, random_state=42)

# total number of pixels in image
print('n features:', n_features)
print('y test:', y_test)

# start testing: train classifiers, attack 

C=3                 # total # of concepts/problems
n_attk_samples = 100 # total # of samples attacked
n_selected = [1,2,3]      # total number of concepts attacked
attk_type = ['l1', 'linf', 'l2']

# test
test_case = TestCase(DATASET, X_train, y_train, X_test, y_test, C)
test_case.train_models()

#display accuracies 
if DATASET == 'mnist':
    label = ['even number', '>= 5', 'zero']
    class_name = 'Digit'
    rotation = 0
elif DATASET == 'fashion_mnist':
    label = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    class_name = 'Clothing'
    rotation = 70
elif DATASET == 'face':
    label = target_names 
    class_name = 'Face'
    rotation = 70

prefix = './eps/'
for atk in attk_type:
    for ns in n_selected:
        # n_start: beginning attack position in the test set
        n_start, attk_mod, adv_x = test_case.attack_models(atk, ns, n_attk_samples);
        original_acc, attacked_acc, original_recall, attacked_recall = test_case.get_metrics(n_start, n_attk_samples, adv_x, C)

        filename = prefix+DATASET+'_'+atk+'_attk'+str(ns)+'_acc.eps'
        plot_acc(original_acc, attacked_acc, C, 'Accuracy', label, class_name, rotation, filename)
        filename = prefix+DATASET+'_'+atk+'_attk'+str(ns)+'_rec.eps'
        plot_acc(original_recall, attacked_recall, C, 'Recall', label, class_name, rotation, filename)

        row = 10 
        #display attacked images
        plt.figure(figsize=(row,row))
        adv_img = np.reshape(adv_x, (-1, 28,28))
        for i in range(row*row):
            plt.subplot(row,row,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(adv_img[i], cmap=plt.cm.binary)
            #plt.xlabel([train_labels[i]])
        filename = prefix+DATASET+'_'+atk+'_attk'+str(ns)+'_imgs.eps'
        plt.savefig(filename, format='eps')
        #plt.show()
        explainer = Explainer()
        filename = prefix+DATASET+'_'+atk+'_attk'+str(ns)+'_shap_'
        explainer.kernel_explain(test_case, X_train, X_test[n_start:n_start+n_attk_samples], adv_x, row, attk_mod, filename)

