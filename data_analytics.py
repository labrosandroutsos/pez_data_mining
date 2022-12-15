import os
from xml.sax.saxutils import prepare_input_source
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import graphviz
from sklearn.model_selection import GridSearchCV
os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/Graphviz/bin"


def train(X_train, y_train, X_test, y_test):

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=2, random_state=42)
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # print("Accurasy is {:2.2%}".format(acc))
    # print("Precision is {:2.2%}".format(precision))
    # print("Recall is {:2.2%}".format(recall))
    # print("F1 is {:2.2%}".format(f1))

    cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    return acc, precision, recall, f1, cm, clf


def tuning(X_train, y_train):
    clf = tree.DecisionTreeClassifier(random_state=42)
    param_grid = {'criterion': ['gini', 'entropy', 'log_loss'], 'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4, 5], 'min_samples_leaf': [1, 2, 3, 4, 5]}
    grid_search_cv = GridSearchCV(clf, param_grid, verbose=1, cv=10, n_jobs= os.cpu_count() - 2)
    grid_result = grid_search_cv.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print('Parameters:')
    for param, value in grid_result.best_params_.items():
        print('\t{}: {}'.format(param, value))

# def preprocessing(X):
#     # df

df = pd.read_csv("clinical_dataset.csv", delimiter=";")
print(df['fried'])
df_nominal = df.select_dtypes(exclude=['number'])
list_nominal = list(df_nominal.columns)
print(list_nominal)
label_encoder = preprocessing.LabelEncoder()
#Oxi automata. Na to kanw me sugkekrimeno tropo gia kathe metavliti
for i in list_nominal:
    print(i)
    df[str(i)] = label_encoder.fit_transform(df[str(i)])
print(df['fried'])
# preprocessing(df)
# df = df.rename(columns={0: "Class", 1: "Alcohol", 2: "Malic_acid", 3: "Ash", 4: "Alcalinity_of_ash", 5: "Magnesium", 6: "Total_phenols", 7: "Flavanoids", 8: "Nonflavanoid_phenols", 9: " Proanthocyanins", 10: "Color_intensity", 11: "Hue", 12: "OD280/OD315_of_diluted wines", 13: "Proline"})
# print(df)

# # Shuffling the dataset
# df = shuffle(df, random_state=42)
# df.reset_index(inplace=True)
# df = df.drop(columns='index')
# # print(df)

# # Check for missing values in the dataset
# missing = df.isna().any().any()
# print("The dataset has missing values: {}".format(missing))

# # Split dataset to input dataset and categorical value
# y = df['Class']
# X = df.drop(columns='Class')

# # Plot the distribution of the classes
# # labels, counts = np.unique(y, return_counts=True)
# # plt.bar(labels, counts, align='center')
# # plt.gca().set_xticks(labels)
# # plt.title("Distribution of Classes")
# # plt.savefig("distribution_of_classes.png")
# # plt.show()

# # Split the dataset to training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

# # Defining metrics lists 
# acc_list = []
# prec_list = []
# rec_list = []
# f1_list = []
# cm_list = []
# clf_list = []

# # Tuning the parameters of the decision tree
# # tuning(X_train, y_train)

# # Train and test the dataset with the decision tree 10 times and take the average of each metric
# for i in range(10):
#     accuracy, precision, recall, f1, cm, clf = train(X_train, y_train, X_test, y_test)
#     acc_list.append(accuracy)
#     prec_list.append(precision)
#     rec_list.append(recall)
#     f1_list.append(f1)
#     cm_list.append(cm)
#     clf_list.append(clf)

# max_acc = max(acc_list)
# max_index = acc_list.index(max_acc)
# print(cm_list[max_index])

# print("Accuracy is {:2.2%} with std {}".format(np.mean(acc_list), np.std(acc_list)))
# print("Precision is {:2.2%} with std {}".format(np.mean(prec_list), np.std(prec_list)))
# print("Recall is {:2.2%} with std {}".format(np.mean(rec_list), np.std(rec_list)))
# print("F1 is {:2.2%} with std {}".format(np.mean(f1_list), np.std(f1_list)))
# clf_best = clf_list[max_index]

# # Visualize the tree
# dot_data = tree.export_graphviz(clf_best, out_file=None, feature_names=X_train.columns.tolist(), class_names=["1", "2", "3"], filled=True, rounded=True)
# graph = graphviz.Source(dot_data)
# graph.format = "jpg"
# graph.render("decision_tree_1")