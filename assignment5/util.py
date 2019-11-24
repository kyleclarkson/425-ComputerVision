import numpy as np
import os
import glob
from sklearn.cluster import KMeans
import time
import pickle

def build_vocabulary(image_paths, vocab_size):
    """ Sample SIFT descriptors, cluster them using k-means, and return the fitted k-means model.
    NOTE: We don't necessarily need to use the entire training dataset. You can use the function
    sample_images() to sample a subset of images, and pass them into this function.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    vocab_size: the number of clusters desired.
    
    Returns
    -------
    kmeans: the fitted k-means clustering model.
    """

    # np.random.seed(123)
    n_image = len(image_paths)

    # Since want to sample tens of thousands of SIFT descriptors from different images, we
    # calculate the number of SIFT descriptors we need to sample from each image.
    n_each = int(np.ceil(10000 / n_image))

    # Initialize an array of features, which will store the sampled descriptors
    # keypoints = np.zeros((n_image * n_each, 2))
    descriptors = np.zeros((n_image * n_each, 128))

    # Iterate over each sample image.
    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',', dtype=float)
        sift_keypoints = features[:, :2]
        sift_descriptors = features[:, 2:]

        # : Randomly sample n_each descriptors from sift_descriptor and store them into descriptors
        # Generate a set of random samples of feature-descriptors and append to list 'descriptors'.
        sample_descriptors = np.random.choice(a=sift_descriptors.shape[0], replace=False)
        descriptors = np.vstack((descriptors, sift_descriptors[sample_descriptors, :]))

    # : prefrom k-means clustering to cluster sampled sift descriptors into vocab_size regions.
    # You can use KMeans from sci-kit learn.
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    # == Save model ==
    print(f"Fitting kmeans vocab")
    stime = time.clock()
    kmeans = KMeans(n_clusters=vocab_size, n_jobs=8).fit(descriptors)
    print(f"Finished fitting kmeans vocab. Time: {time.clock() - stime} Saving model:")
    with open(f'models/kmeans-vocab-{vocab_size}.pkl', 'wb',) as file:
        pickle.dump(kmeans, file)
    return kmeans
    
def get_bags_of_sifts(image_paths, kmeans, mode):
    """ Represent each image as bags of SIFT features histogram.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    kmeans: k-means clustering model with vocab_size centroids.
    mode: For saving either training or test features.

    Returns
    -------
    image_feats: an (n_image, vocab_size) matrix, where each row is a histogram.
    """
    n_image = len(image_paths)
    vocab_size = kmeans.cluster_centers_.shape[0]

    image_feats = np.zeros((n_image, vocab_size))

    # Iterate through each image
    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',', dtype=float)

        # : Assign each feature to the closest cluster center
        # Again, each feature consists of the (x, y) location and the 128-dimensional sift descriptor
        # You can access the sift descriptors part by features[:, 2:]
        # Get matched cluster per each feature
        closest = kmeans.predict(features[:, 2:])

        # : Build a histogram normalized by the number of descriptors
        # Increment the number of features classified per cluster, normalized by number of descriptors.
        num_of_descriptors = features.shape[0]
        for cluster in closest:
            image_feats[i][cluster] += 1 / num_of_descriptors

    # == Save model ==
    print(f"Finished generating BoW from SIFT features. Saving: ")
    save_path = f"models/{mode}-features-{vocab_size}.pkl"
    with open(save_path, 'wb', ) as file:
        pickle.dump(image_feats, file)

    return image_feats

def load(ds_path):
    """ Load from the training/testing dataset.

    Parameters
    ----------
    ds_path: path to the training/testing dataset.
             e.g., sift/train or sift/test 
    
    Returns
    -------
    image_paths: a (n_sample, 1) array that contains the paths to the descriptors. 
    labels: class labels corresponding to each image
    """
    # Grab a list of paths that matches the pathname
    files = glob.glob(os.path.join(ds_path, "*", "*.txt"))
    n_files = len(files)
    image_paths = np.asarray(files)
 
    # Get class labels
    classes = glob.glob(os.path.join(ds_path, "*"))
    labels = np.zeros(n_files)

    for i, path in enumerate(image_paths):
        folder, fn = os.path.split(path)
        labels[i] = np.argwhere(np.core.defchararray.equal(classes, folder))[0,0]

    # Randomize the order
    idx = np.random.choice(n_files, size=n_files, replace=False)
    image_paths = image_paths[idx]
    labels = labels[idx]

    return image_paths, labels


if __name__ == "__main__":
    paths, labels = load("sift/train")
    # build_vocabulary(paths, VOCAB_SIZE)
