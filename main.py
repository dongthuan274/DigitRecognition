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
    
    flat_nearest_vectors = 'data/flat_nearest_vectors'
    chunk_nearest_vectors = 'data/chunk_nearest_vectors'
    histogram_nearest_vectors = 'data/histogram_nearest_vectors'

    process.gen_nearest_k_vectors(combineTest_flat, combineTrain_flat, flat_nearest_vectors, 1000)
    process.gen_nearest_k_vectors(combineTest_chunk, combineTrain_chunk, chunk_nearest_vectors, 1000)
    process.gen_nearest_k_vectors(combineTest_histogram, combineTrain_histogram, histogram_nearest_vectors, 1000)
    
    flat_data = process.load_pkl_file(flat_nearest_vectors)
    chunk_data = process.load_pkl_file(chunk_nearest_vectors)
    histogram_data = process.load_pkl_file(histogram_nearest_vectors)
    
    #test vai cai
    k = 100
    temp = 4124;
    print("predict: ", process.predict_label_using_pkl(flat_data[temp], k),"ril: ", combineTest_flat[temp][1])
    print("predict: ", process.predict_label_using_pkl(chunk_data[temp], k),"ril: ", combineTest_chunk[temp][1])
    print("predict: ", process.predict_label_using_pkl(histogram_data[temp], k),"ril: ", combineTest_histogram[temp][1])

    print("new image predict: ", process.predict_label(combineTest_flat[temp][0],combineTrain_flat, k), "ril: ", combineTest_flat[temp][1])
    print("new image predict: ", process.predict_label(combineTest_chunk[temp][0],combineTrain_chunk, k), "ril: ", combineTest_chunk[temp][1])
    print("new image predict: ", process.predict_label(combineTest_histogram[temp][0],combineTrain_histogram, k), "ril: ", combineTest_histogram[temp][1])

    k_values = range(5, 1000, 5)
    process.graph_accuracy_vs_k_pkl(combineTest_flat, flat_data, k_values)
    process.graph_accuracy_vs_k_pkl(combineTest_chunk, chunk_data, k_values)
    process.graph_accuracy_vs_k_pkl(combineTest_histogram, histogram_data, k_values)
if __name__ == "__main__":
    main()
