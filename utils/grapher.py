import matplotlib.pyplot as plt

if __name__ == '__main__':
    import json

    with open('models/wallace_activation_all-batches_24-04-2021_14-57-02.json') as json_file:
        data = json.load(json_file)

    accuracy = data['accuracy']
    loss = data['loss']

    plt.plot(accuracy,label='train_accuracy')
    plt.plot(loss,label='train_loss', dashes=[6,2])
    plt.title('Results of 10 batches with 10 epochs each')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()