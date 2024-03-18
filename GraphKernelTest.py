from __future__ import print_function
print(__doc__)

from time import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm

from grakel import datasets
from grakel import GraphKernel
from grakel import graph_from_networkx

from matplotlib import pylab as pl

from ExampleGraphs.CreateExampleGraphs import *
import numpy as np

# Loads the Mutag dataset from:
# https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
# the biggest collection of benchmark datasets for graph_kernels.
mutag = datasets.fetch_dataset("MUTAG", verbose=False)
G, y = mutag.data, mutag.target

G = []

number_of_samples = 500

random_num = []

for i in range(0, number_of_samples):
    random_num.append(np.random.randint(10, 1001))
    
print(random_num)

graphs = [circle_graph(random_num[i]) for i in range(0, number_of_samples)] 
graphs += [double_circle(random_num[i]//2, random_num[i] - random_num[i]//2) for i in range(0, number_of_samples)]

for g in graph_from_networkx(graphs, node_labels_tag='abc'):
    G.append(g)

y = []
for i in range(0, number_of_samples):
    y.append(1)
for i in range(0, number_of_samples):
    y.append(-1)

#G = graph_from_networkx(Ex_G, node_labels_tag = "label")
#y = np.array(1)

print(type(G))
print(G)
print(y)


# Train-test split of graph data
G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1)

print(G_train)

start = time()
# Initialise a weifeiler kernel, with a dirac base_kernel.
gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "niter": 2},
                         {"name": "subtree_wl"}], normalize=True)

# Calculate the kernel matrix.
K_train = gk.fit_transform(G_train)
print(K_train)
K_test = gk.transform(G_test)
end = time()

# Initialise an SVM and fit.
clf = svm.SVC(kernel='precomputed', C=2)
clf.fit(K_train, y_train)

# Predict and test.
y_pred = clf.predict(K_test)

# Calculate accuracy of classification.
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", str(round(acc*100, 2)), "% | Took:",
      str(round(end - start, 2)), "s")

fig = pl.figure()
pl.subplot(121)
pl.imshow(K_train)
pl.subplot(122)
pl.imshow(K_test)
pl.show()