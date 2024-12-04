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

    combineTrain_flat = process.combine(train_flat, y_train)
    combineTrain_chunk = process.combine(train_chunk, y_train)
    combineTrain_histogram = process.combine(train_histogram, y_train)
    combineTest_flat = process.combine(test_flat, y_test)
    combineTest_chunk = process.combine(test_chunk, y_test)
    combineTest_histogram = process.combine(test_histogram, y_test)
    
    #test vai cai
    k = 100
    temp = 4124;
    print("predict: ", process.predict_label(combineTest_flat[temp][0], combineTrain_flat, k),"ril: ", combineTest_flat[temp][1])
    print("predict: ", process.predict_label(combineTest_chunk[temp][0], combineTrain_chunk, k),"ril: ", combineTest_chunk[temp][1])
    print("predict: ", process.predict_label(combineTest_histogram[temp][0], combineTrain_histogram, k),"ril: ", combineTest_histogram[temp][1])
if __name__ == "__main__":
    main()
