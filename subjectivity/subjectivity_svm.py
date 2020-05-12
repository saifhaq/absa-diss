from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np 
import os.path as path
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt 


def plot_cm(cm,
            target_names,
            title='Confusion matrix',
            cmap=None,
            normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    thresh = 0.9

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def stoplist(file_name = "stopwords.txt"):
    """
        Returns an array of stopwords, from each line of the 
        *file_name* text file
    """
    stopwords_txt = open(path.join('preprocessing', file_name))
    stoplist = []
    for line in stopwords_txt:
        values = line.split()
        stoplist.append(values[0])
    stopwords_txt.close()
    return stoplist
    
def print_stats(y_pred, y_test, model_name):
    """
        Helper function using data from Tensorflow's model evaluation
        function to return the F1 and print model performance stats. 
        Also updates data_df df to contain model acc and f1
    """

    data_df = pd.read_pickle(path.join('subjectivity', path.join('results', 'data_df.pkl')))
    test_f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_pred, y_test)
    try:
        best_f1 = data_df[data_df['model']==model_name]['f1'].values[0]
        best_f1 = 0 
    except: 
        best_f1 = 0 

    if test_f1 > best_f1:
        best_f1 = test_f1   
        data_df = data_df[data_df.model != model_name]
        data_df = data_df.append({'model': model_name, 'acc': acc, 'f1': test_f1}, ignore_index=True)
        
    data_df.to_pickle(path.join('subjectivity', path.join('results', 'data_df.pkl')))
    print(data_df)

    print('Test Accuracy: {}'.format(acc))
    print('---------------')
    print('Test Precision: {}'.format(precision_score(y_test, y_pred, average="macro")))
    print('Test Recall: {}'.format(recall_score(y_test, y_pred, average="macro")))
    print('---------------')
    print('Test F1: {}'.format(test_f1))
    return test_f1

train_df = pd.read_pickle(path.join('subjectivity', path.join('pandas_data', 'TRAIN_SUBJECTIVITY.pkl')))
test_df = pd.read_pickle(path.join('subjectivity', path.join('pandas_data', 'TEST_SUBJECTIVITY.pkl')))

stopwords = stoplist()
train_df['text'] = train_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))
test_df['text'] = test_df['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))

x_train, y_train = train_df.text, train_df.subjectivity
x_test, y_test = test_df.text, test_df.subjectivity

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier()),    
     ])

text_clf.fit(x_train, y_train)

predicted = text_clf.predict(x_test)

class_names = ["Objective", "Subjective"]
print_stats(y_test, predicted, 'svm')
cm = confusion_matrix(y_test, predicted)

title = "SVM Model Subjectivity Normalized Confusion Matrix"
plot_cm(cm, class_names, title=title)

