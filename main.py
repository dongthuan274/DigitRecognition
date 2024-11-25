import process

def main():
    x_train, y_train = process.load_mnist("data/", kind="train")
    x_test, y_test = process.load_mnist("data/", kind="t10k")

    print(f"training images: {x_train.shape}")
    print(f"training labels: {y_train.shape}")

    print(f"test images: {x_test.shape}")
    print(f"test labels: {y_test.shape}")

if __name__ == "__main__":
    main()
