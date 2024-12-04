import process

def main():
    x_train, y_train = process.load_mnist("data/", kind="train")
    x_test, y_test = process.load_mnist("data/", kind="t10k")

    train_chunk = process.chunk_matrixize(x_train)
    test_chunk = process.chunk_matrixize(x_test)

    print(f"training images: {x_train.shape}")
    print(f"training labels: {y_train.shape}")

    print(f"test images: {x_test.shape}")
    print(f"test labels: {y_test.shape}")

    print(f"training images after flattening by chunk: {train_chunk.shape}")
    print(f"test images after flattening by chunk: {test_chunk.shape}")

if __name__ == "__main__":
    main()
