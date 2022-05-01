import torch
from BasicNet import train_basic_net
import math


def generate_dataset(data_file):
    test_data = []
    with open(data_file, 'r') as in_file:
        for line in in_file:
            split_line = line.strip().split(',')
            to_tensor = list(map(float, split_line[1:]))
            label = int(split_line[0])
            test_data.append((to_tensor, label))
    in_file.close()
    return test_data


def extract_folds(data_list, folds: int):
    fold_len = math.ceil(len(data_list) / folds)
    segments = []
    index = 0
    while index < len(data_list) - fold_len:
        segments.append(data_list[index:int(index + fold_len)])
        index = int(index + fold_len)
    segments.append(data_list[index:])
    return segments


def cross_validate(training_data_file, folds: int):
    data = generate_dataset(training_data_file)
    data = list(map(convert_example, data))
    segments = extract_folds(data, folds)
    for i in range(len(segments) - 1):
        test_data = segments.pop(0)
        trained = train_basic_net(segments)
        correct = 0
        for item in test_data:
            inp, label = item
            if trained.predict(inp) == label:
                correct += 1
        print("Test " + str(i) + " accuracy: " + str(correct / len(test_data)))
        segments.append(test_data)


# TODO: ask about when I should be checking the accuracy for these ^^^


def convert_example(ex):
    return torch.tensor(ex[0]), ex[1]


def main():
    # debugging_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # print(extract_folds(debugging_data, 2))

    cross_validate('mnist_train_0_4.csv', 8)

    # trained = train_basic_net(data)
    # test = generate_dataset('mnist_train_0_4.csv')
    # test = list(map(convert_example, test))
    # correct = 0
    # for ex in test:
    #     inp, label = ex
    #     if trained.predict(inp) == label:
    #         correct += 1
    # print("Accuracy:", correct / len(test))


if __name__ == '__main__':
    main()
