import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from plot_bar import plot_shap




class Explainer():
    def __init__(self):
        pass

    def explain (self, dataset_orig_train, dataset_orig_test, mi_type):

        model = LogisticRegression(penalty="l2", C=0.1)
        model.fit(dataset_orig_train.features, dataset_orig_train.labels.ravel())

        explainer = shap.LinearExplainer(model, dataset_orig_train.features, feature_dependence="independent")
        shap_values = explainer.shap_values(dataset_orig_test.features)
        X_test_array = dataset_orig_test.features # we need to pass a dense version for the plotting functions
        shap.summary_plot(shap_values, X_test_array, dataset_orig_train.feature_names)
        #shap.plots.scatter(shap_values[:,"race"], color=shap_values)
        plt.show()
        plt.savefig('./eps/summary_plot_'+mi_type+'.jpg')


    # test_case object contains: classifiers (sgd_clf), concepts (problems)
    # x_train: raw input, will be flattened 
    # adv_x: should be flattened as input 
    # row: total number of rows/columns of the subplots  
    def kernel_explain(self, test_case, x_train, x, adv_x, row, attk_mod, filename):
        models = test_case.sgd_clf
        problems = test_case.problems
        attk_problems = [problems[i] for i in attk_mod]
        num_inst = x.shape[0]
        n_features = x_train.shape[1]*x_train.shape[2]
        X_train = np.reshape(x_train, (-1, n_features))

        for j in range(len(models)):
            adv_shap = []
            x_shap=[]
            for i in range(num_inst):
                # explain all the predictions in the test set
                # sample 1000 from the training to speed up
                explainer = shap.KernelExplainer(models[j].predict_proba, shap.sample(X_train, 1000))
                shap_values = explainer.shap_values(x[i].flatten())
                x_shap.append(shap_values[1])
                adv_shap_values = explainer.shap_values(adv_x[i])
                adv_shap.append(adv_shap_values[1])
            plot_shap(x_shap, x_train.shape[1], x_train.shape[2], row, problems[j], attk_problems, False, filename)
            plot_shap(adv_shap, x_train.shape[1], x_train.shape[2], row, problems[j], attk_problems, True, filename)
