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


x_train, y_train = load_mnist("data/", kind="train")
x_test, y_test = load_mnist("data/", kind="t10k")

print(f"training images: {x_train.shape}")
print(f"training labels: {y_train.shape}")

print(f"test images: {x_test.shape}")
print(f"test labels: {y_test.shape}")
