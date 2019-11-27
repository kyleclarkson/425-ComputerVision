import numpy as np
import os
import glob
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import time
import pickle
import matplotlib.pyplot as plt

np.random.seed(123)
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

    # : preform k-means clustering to cluster sampled sift descriptors into vocab_size regions.
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

    folder_name = ""
    for i, path in enumerate(image_paths):
        folder, fn = os.path.split(path)
        labels[i] = np.argwhere(np.core.defchararray.equal(classes, folder))[0, 0]
        # if folder_name != folder:
        #     folder_name = folder
        #     print(f"Folder: {folder_name}, index: = {labels[i]}")

    # Randomize the order
    idx = np.random.choice(n_files, size=n_files, replace=False)
    image_paths = image_paths[idx]
    labels = labels[idx]

    return image_paths, labels

def generate_histogram(image_feats,
                       labels,
                       vocab_size):
    print(f"Generating histograms:")
    # Mapping from class label [0,14] to class name.
    class_dict = {0: "Bedroom",
                  1: "Coast",
                  2: "Forest",
                  3: "Highway",
                  4: "Industrial",
                  5: "InsideCity",
                  6: "Kitchen",
                  7: "LivingRoom",
                  8: "Mountain",
                  9: "Office",
                  10: "OpenCountry",
                  11: "Store",
                  12: "Street",
                  13: "Suburb",
                  14: "TallBuilding"}

    # A dictonary of histograms with K/V pairs:
    # key: class name (above)
    # value: pair (vector with length corresponding to number of features, total count of features found)
    histograms = {}
    number_of_feats = image_feats.shape[1]
    # Increment histograms per class.
    for idx, label in enumerate(labels):
        class_name = class_dict.get(label)
        class_histogram, count = histograms.get(class_name, (np.zeros((1, number_of_feats)), 0))
        # Update histogram
        histograms[class_name] = (np.add(class_histogram, image_feats[idx]), count+1)

    # Compute average histogram per class and display.
    for class_name, (class_histogram, count) in histograms.items():
        plt.bar(np.arange(number_of_feats), np.divide(class_histogram, count)[0])
        plt.title(f"Histogram-{class_name}")
        plt.xlabel("Number of features in BoW Representation")
        plt.savefig(f"histograms/{class_name}.jpg")
        plt.close()

def generate_confusion_matrix(pred_labels, true_labels, title, vocab_size):

    # Mapping from class label [0,14] to class name.
    class_dict = {0: "Bedroom",
                  1: "Coast",
                  2: "Forest",
                  3: "Highway",
                  4: "Industrial",
                  5: "InsideCity",
                  6: "Kitchen",
                  7: "LivingRoom",
                  8: "Mountain",
                  9: "Office",
                  10: "OpenCountry",
                  11: "Store",
                  12: "Street",
                  13: "Suburb",
                  14: "TallBuilding"}
    # Generate normalized confusion matrix.
    class_names = class_dict.values()
    cm = confusion_matrix(true_labels, pred_labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Show plot with appropriate labels.
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        title=title,
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names
    )

    # Rotate x label ticks for viewing.
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor")

    plt.xlabel = "Predicted label"
    plt.ylabel = "True label"
    plt.tight_layout()
    # Save image
    plt.show()
    is_saving = True
    if is_saving:
        plt.savefig(f"confusion-matrices/{title}.jpg")

if __name__ == "__main__":
    paths, labels = load("sift/train")
    # build_vocabulary(paths, VOCAB_SIZE)
