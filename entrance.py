import tensorflow as tf
import matplotlib.pyplot as plt
from data.Titanic_kaggle.data_process import load_data


def load_model(model_name):
    if model_name == "LR":
        from models.LR import construct_model
    elif model_name == "LR":
        from models.FM import construct_model
    else:
        print ("The {} is not support currently, using LR for instead".format(model_name))
        from models.LR import construct_model
    return construct_model

def main(model_name):
    x_train, x_test, y_train, y_test = load_data()
    n_sample, n_feature = x_train.shape

    construct_model = load_model(model_name.upper())
    model_layer = construct_model(n_feature)

    model = tf.keras.Sequential()
    model.add(model_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #               loss=tf.keras.losses.binary_crossentropy,
    #               metrics=[tf.keras.metrics.binary_accuracy])

    history = model.fit(x_train, y_train, epochs=100, validation_split=0.2)
    print (history)

    #learning curves in validation dataset
    plt.plot(history.epoch, history.history["val_loss"])

    #evaluate model on test dataset
    model.evaluate(x_test, y_test)

    #pick up varaibles in layers
    # variables = model.layers[0].variables
    # w = variables[0].numpy()
    # b = variables[1].numpy()


if __name__ == '__main__':
    model_name = "FM"
    main(model_name)