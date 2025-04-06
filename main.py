'''
    <START SET UP>
    Suppress warnings and import necessary libraries.
    Import code for loading data and extracting features.
'''

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import math 

'''
    <END SET UP>
'''

'''
    Load facial landmarks (5 or 68)
'''

X = np.load("C:\\Users\\clutc\\Desktop\\py\\X-68-Caltech.npy")
y = np.load("C:\\Users\\clutc\\Desktop\\py\\y-68-Caltech.npy")
num_identities = y.shape[0]

'''
    Transform landmarks into features
'''

features = []
for k in range(num_identities):
    person_k = X[k]
    features_k = []
    for i in range(person_k.shape[0]):
        for j in range(person_k.shape[0]):
            p1 = person_k[i,:]
            p2 = person_k[j,:]      
            features_k.append(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) #manhattan distance formula can used euclidean as well
    features.append(features_k)
features = np.array(features)

''' 
    Create an instance of the classifier
'''

#53%
from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier()

#42%
from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier()

#61%
# k-Nearest Neighbors - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
#clf = model = KNeighborsClassifier(n_neighbors=5, metric='cosine', weights='uniform')

#0% maybe i must have done something wrong
from sklearn.naive_bayes import BernoulliNB
#clf = BernoulliNB() 

#85%
from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB() 

#78%
from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB() 

#86%
from sklearn.naive_bayes import ComplementNB
clf = ComplementNB(alpha=0.1)  







num_correct = 0
num_incorrect = 0

for i in range(0, len(y)):
    query_X = features[i, :]
    query_y = y[i]
    
    template_X = np.delete(features, i, 0)
    template_y = np.delete(y, i)
        
    # Set the appropriate labels
    # 1 is genuine, 0 is impostor
    y_hat = np.zeros(len(template_y))
    y_hat[template_y == query_y] = 1 
    y_hat[template_y != query_y] = 0
    
    # Train the classifier
    clf.fit(template_X, y_hat) 
    
    # Predict the label of the query
    y_pred = clf.predict(query_X.reshape(1,-1)) 
    
    # Get results
    if y_pred == 1:
        num_correct += 1
    else:
        num_incorrect += 1

# Print results
print(clf)
print()
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect))) 

import matplotlib.pyplot as plt
algorithms = ["cosine", "jaccard", "euclidean", "manhattan", "canberra"]
accuracies = [62, 1, 57, 55, 63]
plt.bar(algorithms, accuracies, color='skyblue')
plt.ylabel('Accuracy (%)')
plt.xlabel('M (metric)')
plt.title('Parameters (p=(n_neighbors=5, metric=M, weights=distance))')
plt.show()




