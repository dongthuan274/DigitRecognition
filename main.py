import process

def main():
    x_train, y_train = process.load_mnist("data/", kind="train")
    x_test, y_test = process.load_mnist("data/", kind="t10k")

    train_flat, train_chunk, train_histogram = process.extract_features(x_train)
    test_flat, test_chunk, test_histogram = process.extract_features(x_test)

    print(f"training images: {x_train.shape}\t\ttest images: {x_test.shape}")
    print(f"train labels: {y_train.shape}\t\t\ttest labels: {y_test.shape}")

    print(f"FLAT training images: {train_flat.shape}\t\tFLAT test images: {test_flat.shape}")

    print(f"CHUNK training images: {train_chunk.shape}\t\tCHUNK test images: {test_chunk.shape}")

    print(f"HISTOGRAM training images: {train_histogram.shape}\t\tHISTOGRAM test images: {test_histogram.shape}")

if __name__ == "__main__":
    main()
