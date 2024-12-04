import gzip
import os
import numpy as np

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

<<<<<<< HEAD
def chunk_flattening(images):
    chunk_matrix = []
    rows_per_chunk, cols_per_chunk = 4, 4
=======
def chunk_vectorize(images, rows_per_chunk = 4, cols_per_chunk = 4):
    '''
    reshape() image matrix (2D) to 4D
    mean() to calculate average value of every chunk
    '''
    chunk_vectors = []
>>>>>>> 5e380ad7c8cc3a1507131a8403430e6011e3a0e3

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
<<<<<<< HEAD
        chunk_matrix.append(chunk_vector)
    
    return np.array(chunk_matrix)
=======
        chunk_vectors.append(chunk_vector)

    return np.array(chunk_vectors)
>>>>>>> 5e380ad7c8cc3a1507131a8403430e6011e3a0e3

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
<<<<<<< HEAD
    flat_vector = flat_vectorize(images)
    chunk_matrix = chunk_flattening(images)
    histogram_vector = histogram_vectorize(images)

    return flat_vector, chunk_matrix, histogram_vector
=======
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
>>>>>>> 5e380ad7c8cc3a1507131a8403430e6011e3a0e3
