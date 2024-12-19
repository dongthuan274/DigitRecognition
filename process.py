import gzip
import os
import numpy as np
from matplotlib import pyplot as plt

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

def vectorize(images):
    '''
    flatten() convert image matrix (2D) to vector (1D)
    '''
    flat_vectors = [image.flatten() / 255.0 for image in images]
    return np.array(flat_vectors)

def sampling(images, rows_per_chunk = 4, cols_per_chunk = 4):
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

def histogram(images):
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
    flat_vectors = vectorize(images)
    chunk_vectors = sampling(images)
    histogram_vectors = histogram(images)

    return flat_vectors, chunk_vectors, histogram_vectors

def combine(features, labels):
    combined_list = []
    for feature, label in zip(features, labels):
        combined_list.append([feature, label])
    return combined_list

