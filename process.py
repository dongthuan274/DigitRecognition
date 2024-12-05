import gzip
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

IMG_SIZE = 28

def load_mnist(path, kind="train"):
    label_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    image_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(label_path, "rb") as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)

    with gzip.open(image_path, "rb") as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = (
            np.frombuffer(buffer, dtype=np.uint8)
            .reshape(len(labels), IMG_SIZE, IMG_SIZE)
            .astype(np.float64)
        )

    return images, labels

def flat_vectorize(images):
    '''
    flatten() convert image matrix (2D) to vector (1D) using
    '''
    flat_vectors = [image.flatten() / 255.0 for image in images]
    return np.array(flat_vectors)

def chunk_vectorize(images, rows_per_chunk = 4, cols_per_chunk = 4):
    '''
    reshape() image matrix (2D) to 4D
    mean() to calculate average value of every chunk
    '''
    chunk_vectors = []

    for image in images:
        chunk_vector = (
            (image.reshape(
                image.shape[0] // rows_per_chunk,
                rows_per_chunk,
                image.shape[1] // cols_per_chunk,
                cols_per_chunk
            )
            .mean(axis=(1, 3))
            / 255.0
            )
            .flatten()
        )
        chunk_vectors.append(chunk_vector)

    return np.array(chunk_vectors)

def histogram_vectorize(images):
    '''
    numsbin = number of blocks
    histogram() calculate frequency of element in 1D array
    '''
    nums_bin = 256
    histogram_vectors = []

    for image in images:
        flat_image = image.flatten()
        histogram_vector, ignore = np.histogram(flat_image, bins = nums_bin, range = (0, 255))
        histogram_vector = histogram_vector / (IMG_SIZE*IMG_SIZE)
        histogram_vectors.append(histogram_vector)

    return np.array(histogram_vectors)

def extract_features(images):
    flat_vectors = flat_vectorize(images)
    chunk_vectors = chunk_vectorize(images)
    histogram_vectors = histogram_vectorize(images)

    return flat_vectors, chunk_vectors, histogram_vectors

def calcDist(vectorized_image1, vectorized_image2):
    return np.sqrt(np.sum((vectorized_image1 - vectorized_image2) ** 2))

def combine(vectorized_images, labels):
    combineList = []
    for vectorized_image, label in zip(vectorized_images, labels):
        combineList.append([vectorized_image, label])
    return combineList

def gen_nearest_k_vectors(combineTest_images, combineTrain_images, output_file, k=1000):
    if os.path.exists(output_file):
        return
    nearest_neighbors = []
    for test_image, test_label in combineTest_images:
        distances = []
        for train_image, train_label in combineTrain_images:
            dist = calcDist(test_image, train_image)
            distances.append((dist, train_label))
        distances.sort(key=lambda x: x[0])
        nearest_labels = [train_label for ignore, train_label in distances[:k]]
        nearest_neighbors.append(nearest_labels)
    with open(output_file, 'wb') as f:
        pickle.dump(nearest_neighbors, f)

def load_pkl_file(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data

def predict_label_using_pkl(nearest_labels, k):
    k_nearest = nearest_labels[:k]
    return np.bincount(k_nearest).argmax()

def predict_label(vectorized_image, combineTestImages, k):
    distances = []
    for i in combineTestImages:
        dist = calcDist(vectorized_image, i[0])
        distances.append([dist, i[1]])
    distances.sort(key=lambda x: x[0])
    nearest_distances = distances[:k]
    nearest_labels = [label[1] for label in nearest_distances]
    most_common_label = max(nearest_labels, key=nearest_labels.count)
    return most_common_label

def graph_accuracy_vs_k_pkl(combineTest_images, nearest_neighbors, k_values):
    accuracies = []
    true_labels = np.array([label for ignore, label in combineTest_images])
    for k in k_values:
        predictions = []
        for nearest_labels in nearest_neighbors:
            predicted_label = predict_label_using_pkl(nearest_labels, k)
            predictions.append(predicted_label)
        predictions = np.array(predictions)
        correct_predictions = np.sum(predictions == true_labels)
        accuracy = correct_predictions / len(combineTest_images)
        accuracies.append(accuracy)
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', color='b', label='Accuracy')
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k in k-Nearest Neighbors")
    plt.grid()
    plt.legend()
    plt.show()
