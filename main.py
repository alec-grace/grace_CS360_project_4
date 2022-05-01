import torch
from BasicNet import train_basic_net
import math
from time import perf_counter


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
    most_accurate = [0, 0]
    for neural_net in range(3):
        accuracies = []
        for i in range(len(segments)):
            test_data = segments.pop(0)
            training_data = []
            for set_list in segments:
                for individual in set_list:
                    training_data.append(individual)
            trained = train_basic_net(training_data, neural_net + 1)
            correct = 0
            for item in test_data:
                inp, label = item
                if trained.predict(inp) == label:
                    correct += 1
            accuracies.append(correct / len(test_data))
            segments.append(test_data)
        accuracy = sum(accuracies) / len(accuracies)
        if accuracy > most_accurate[1]:
            most_accurate = [neural_net + 1, accuracy]
        print('NeuralNet' + str(neural_net + 1) + ' average accuracy of:', accuracy)
    return most_accurate[0]


def convert_example(ex):
    return torch.tensor(ex[0]), ex[1]


def main():

    start = perf_counter()
    best_net = cross_validate('mnist_train_0_4.csv', 4)
    final_data = generate_dataset('mnist_train_0_4.csv')
    final_data = list(map(convert_example, final_data))
    final_net = train_basic_net(final_data, best_net)
    test = generate_dataset('mnist_test_0_4.csv')
    test_data = list(map(convert_example, test))
    correct = 0
    for number in test_data:
        inp, label = number
        if final_net.predict(inp) == label:
            correct += 1
    end = perf_counter()
    print('\nFinal accuracy, using NeuralNet' + str(best_net) + ': ', correct / len(test_data))
    total = end - start
    print('It took', total / 60, 'minutes to complete this program...\n\nYikes.')


if __name__ == '__main__':
    main()
