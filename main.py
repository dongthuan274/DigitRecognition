import process

def main():
    x_train, y_train = process.load_mnist("data/", kind="train")
    x_test, y_test = process.load_mnist("data/", kind="t10k")

    train_flat, train_chunk, train_histogram = process.extract_features(x_train)
    test_flat, test_chunk, test_histogram = process.extract_features(x_test)

    print(f"training images: {x_train.shape}")
    print(f"training labels: {y_train.shape}")

    print(f"test images: {x_test.shape}")
    print(f"test labels: {y_test.shape}")

    print(f"training images after flattening by chunk: {train_chunk.shape}")
    print(f"test images after flattening by chunk: {test_chunk.shape}")

    print(f"training images after histogram vectorization: {train_histogram.shape}")
    print(f"test images after histogram vectorization: {test_histogram.shape}")
if __name__ == "__main__":
    main()
