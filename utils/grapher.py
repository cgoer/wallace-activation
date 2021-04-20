import matplotlib.pyplot as plt

if __name__ == '__main__':
    import json

    with open('wallace_activation_batch1_12-04-2021_23-18-37.json') as json_file:
        data = json.load(json_file)

    accuracy = data['accuracy']
    loss = data['loss']

    plt.plot(accuracy,label='train_accuracy')
    plt.plot(loss,label='train_loss', dashes=[6,2])
    plt.title('Results of Batch 2 / 10')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()