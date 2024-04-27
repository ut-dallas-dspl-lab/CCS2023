import numpy as np
import matplotlib.pyplot as plt

def plot_acc(original, attacked, n_groups, ylabel, label, class_name, rotation, filename):
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.7

    rects1 = plt.bar(index, original, bar_width,
    alpha=opacity,
    color='b',
    label='Original')

    rects2 = plt.bar(index + bar_width, attacked, bar_width,
    alpha=opacity,
    color='r',
    label='Attacked')

    plt.xlabel(class_name)
    plt.ylabel(ylabel)
    plt.title(ylabel + ' before/after attacks')
    plt.xticks(index + bar_width, label, rotation = rotation)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(filename, format='eps')
    #plt.show()

# plot shapley values
# x_shap: Shapley values in flattened format
# sa,sb: width, height of the images
# row: number of subplots in a row 
# prob: problem (classifier name)
# attk_prob: problems attacked (classifiers attacked)
# filename: prefix of the saved file
 
def plot_shap(x_shap, sa, sb, row, prob, attk_prob, attk_bool, filename):
    #display attacked images
    x = np.array(x_shap)
    plt.figure(figsize=(row,row))
    img = np.reshape(x, (-1, sa,sb))
    if attk_bool:
        filename = filename+'Model_'+prob+'_Attk_'+''.join(attk_prob)+'.eps'
        plt.suptitle('Model: '+prob+', Attack: ' + ''.join(attk_prob))
    else:
        filename = filename+'Model_'+prob+'.eps'
        plt.suptitle('Model: '+prob)
    for i in range(row*row):
        plt.subplot(row,row,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img[i], cmap=plt.cm.binary)
    plt.savefig(filename, format='eps')
    #plt.show()

