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
    flat_vectors = [image.flatten() / 255.0 for image in images]
    return np.array(flat_vectors)

def chunk_flattening(images):
    chunk_matrix = []
    rows_per_chunk, cols_per_chunk = 4, 4

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
        chunk_matrix.append(chunk_vector)
    
    return np.array(chunk_matrix)

def histogram_vectorize(images):
    nums_bin = 256
    histograms = []
    
    for image in images:
        flat_image = image.flatten()
        histogram, ignore = np.histogram(flat_image, bins = nums_bin, range = (0, 255))
        histogram = histogram / (IMG_SIZE*IMG_SIZE)
        histograms.append(histogram)
        
    return np.array(histograms)

def extract_features(images):
    flat_vector = flat_vectorize(images)
    chunk_matrix = chunk_flattening(images)
    histogram_vector = histogram_vectorize(images)

    return flat_vector, chunk_matrix, histogram_vector
