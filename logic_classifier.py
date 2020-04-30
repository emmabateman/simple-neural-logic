import numpy as np
import matplotlib.pyplot as plt
from nn_class import NeuralNet

class_labels = ["AND", "OR", "XOR", "NAND", "NOR", "EQUAL"]

def evaluate(predictions, gold):
    predictions = [np.around(p).astype(int) for p in predictions]
    gold = [np.around(g).astype(int) for g in gold]

    true_negative = np.zeros(len(class_labels))
    true_positive = np.zeros(len(class_labels))
    false_negative = np.zeros(len(class_labels))
    false_positive = np.zeros(len(class_labels))

    for i in range(len(predictions)):
        true_positive = true_positive + np.logical_and(predictions[i], gold[i])
        true_negative = true_negative + np.logical_not(np.logical_or(predictions[i], gold[i]))
        false_positive = false_positive + np.logical_and(np.logical_not(gold[i]), predictions[i])
        false_negative = false_negative + np.logical_and(gold[i], np.logical_not(predictions[i]))

    precision = np.nan_to_num(true_positive / (true_positive + false_positive))
    recall = np.nan_to_num(true_positive / (true_positive + false_negative))
    f1 = np.nan_to_num(2 * (precision*recall) / (precision+recall)
)
#    for i in range(len(class_labels)):
#        print(class_labels[i])
#        print("\tF1: {0}".format(f1[0,i]))
#        print("\tR: {0}".format(recall[0,i]))
#        print("\tP: {0}".format(precision[0,i]))

    return f1, recall, precision

def logic_data():
    data = []
    for i in range(4):
        x = int(int(i/2)%2 == 1)
        y = int(i%2 == 1)
        inputs = np.array([[x, y]])
        labels = np.array([[x & y, x | y, x ^ y, not (x & y), not (x | y), x == y]])
        data.append((inputs, labels))
    return data

net = NeuralNet(2, 6, [10, 10], rate=0.1)
data = logic_data()
inputs = [d[0] for d in data]
labels = [d[1] for d in data]

f1_results = np.zeros((len(class_labels), 0))
for trial in range(100):
    print("testing with {0} epochs".format(trial+1))
    net.train(data, epochs=trial+1, cooling=0.99)
    f1, recall, precision = evaluate([net.forward(i) for i in inputs], labels)
    f1_results = np.append(f1_results, f1.transpose(), 1)

legend = []
for i in range(f1_results.shape[0]):
    plt.plot(f1_results[i], linestyle='dashed')
    legend.append(class_labels[i])

plt.legend(legend)
plt.xlabel("epochs")
plt.ylabel("f1 score")
plt.show()
