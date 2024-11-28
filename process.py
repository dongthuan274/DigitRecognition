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

def chunk_vectorize(images):
    pass

def histogram_vectorize(images):
    pass

def extract_features(images):
    flat_vector = flat_vectorize(images)
    chunk_vector = chunk_vectorize(images)
    histogram_vector = histogram_vectorize(images)

    return flat_vector, chunk_vector, histogram_vector
