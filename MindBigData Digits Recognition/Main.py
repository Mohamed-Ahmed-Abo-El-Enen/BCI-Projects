import LoadData

if __name__ == "__main__":
    dataset = LoadData.MindBigData()
    samples = dataset.get_dataset()

    label = 1
    specific_label_dataset_1 = dataset.get_specific_label(samples, label)
    label = 2
    specific_label_dataset_2 = dataset.get_specific_label(samples, label)

    max_shape = list()
    max_shape.append(dataset.get_max_feature_vector_length(specific_label_dataset_1))
    print(max_shape[-1])

    max_shape.append(dataset.get_max_feature_vector_length(specific_label_dataset_2))
    print(max_shape[-1])

    pre_max_padding = max(max_shape)
    label = 1
    dataset.save_labe_data_file(specific_label_dataset_1, label, pre_max_padding)
    label = 2
    dataset.save_labe_data_file(specific_label_dataset_2, label, pre_max_padding)

    dataset.read_txt_data_file('features_dataset_for_1.txt')
