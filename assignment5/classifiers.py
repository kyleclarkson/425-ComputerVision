 #Starter code prepared by Borna Ghotbi for computer vision
 #based on MATLAB code by James Hay

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import multiclass, svm


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.

    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector
        indicating the predicted category for each test image.

    Usefull funtion:

    	# You can use knn from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    '''

    # Number of neighbours to use.
    nn = KNeighborsClassifier(n_neighbors=k, n_jobs=8)
    training_error = nn.fit(train_image_feats, train_labels).score(train_image_feats, train_labels)
    print(f"KNN training error: {training_error:.4f}")
    predicted_labels = nn.predict(test_image_feats)
    return predicted_labels

'''This function will predict the category for every test image by finding
the training image with most similar features. Instead of 1 nearest
neighbor, you can vote based on k nearest neighbors which will increase
performance (although you need to pick a reasonable value for k). '''



'''This function will train a linear SVM for every category (i.e. one vs all)
and then use the learned linear classifiers to predict the category of
very test image. Every test feature will be evaluated with all 15 SVMs
and the most confident SVM will "win". Confidence, or distance from the
margin, is W*X + B where '*' is the inner product or dot product and W and
B are the learned hyperplane parameters. '''

def svm_classify(train_image_feats, train_labels, test_image_feats, c):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string 
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.
        
    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector 
        indicating the predicted category for each test image.

    Usefull funtion:
    	
    	# You can use svm from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/svm.html

    '''

    clf = svm.LinearSVC(
        penalty='l2',
        dual=False,
        C=c
        )
    training_error = clf.fit(train_image_feats, train_labels).score(train_image_feats, train_labels)
    print(f"SVM training error: {training_error:.4f}")
    predicted_labels = clf.predict(test_image_feats)
    return predicted_labels

