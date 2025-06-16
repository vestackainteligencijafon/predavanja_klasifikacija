def compute_basic_eval_measures(cm, model_name):

    from pandas import Series

    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]

    accuracy = (TP + TN)/cm.sum()
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    F1 = 2*precision*recall/(precision + recall)

    return Series(data=[accuracy, precision, recall, F1],
                  index=['accuracy', 'precision', 'recall', 'F1'],
                  name=model_name)


def plot_confusion_matrix(cm, cls_labels):

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=cls_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.show()