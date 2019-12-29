from decisionTree.DecisionTree import *
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import numpy as np

# IRIS DataSet
header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
df = pd.read_csv('data/iris.data', header=None, names=header)

# DataSet Downloaded from UCI Machine Learning dataset repository
# Download dataset from the link https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/
# To run with the above downloaded dataset, uncomment next two lines

#df = pd.read_csv('data/breast-cancer-2.csv')
#header = list(df.columns)
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

test = lst[0]
lab = classify(test, t)
test = lst[0:10]
print("Accuracy = " + str(computeAccuracy(test, t)))


trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
test_acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(test_acc))

innerNodes = getInnerNodes(t)
pruned_list = list()
pruned_accuracy = 0
for node in innerNodes:
    t = build_tree(train, header)
    if node.id != 0:
        pruned_list.append(node.id)
        t_pruned = prune_tree(t, pruned_list)
        pruned_accuracy = computeAccuracy(test, t_pruned)
        if pruned_accuracy > test_acc:
            test_acc = pruned_accuracy
        pruned_list.remove(node.id)
print("*************Tree after pruning*******")
print("Accuracy on test = " + str(test_acc))

# t_pruned = prune_tree(t, [26, 11, 5])
#
# print("*************Tree after pruning*******")
# print_tree(t_pruned)
# acc = computeAccuracy(test, t)
# print("Accuracy on test = " + str(acc))
