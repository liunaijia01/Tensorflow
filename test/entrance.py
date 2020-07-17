import tensorflow as tf
import matplotlib.pyplot as plt
#from data.Titanic_kaggle.data_process import load_data
from data.books_ctr.data_process import load_data
from sklearn.metrics import roc_auc_score


def load_model(model_name):
    if model_name == "LR":
        from algorithm.LR import construct_model
    elif model_name == "FM":
        from algorithm.FM import construct_model
    else:
        print ("The {} is not support currently, using LR for instead".format(model_name))
        from algorithm.LR import construct_model
    return construct_model

def main(model_name):
    x_train, x_test, y_train, y_test = load_data()
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(512)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(512)
    n_sample, n_feature = x_train.shape

    construct_model = load_model(model_name.upper())
    model_layer = construct_model(n_feature)

    model = tf.keras.Sequential()
    model.add(model_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #               loss=tf.keras.losses.binary_crossentropy,
    #               metrics=[tf.keras.metrics.binary_accuracy])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=1e-3)]
    #history = model.fit(x_train, y_train, epochs=10, validation_split=0.5, callbacks=callbacks)
    history = model.fit(train_ds, epochs=100, validation_freq=1, callbacks=callbacks)
    #print (history.history)

    #learning curves in validation dataset
    plt.plot(history.epoch, history.history["loss"])
    plt.show()

    #evaluate model on basic_usage dataset
    #model.evaluate(x_test, y_test)
    model.evaluate(test_ds)

    auc_test = roc_auc_score(y_test, model(x_test))
    print ("AUC for basic_usage dataset is {}".format(auc_test))

    #pick up varaibles in layers
    # variables = model.layers[0].variables
    # w = variables[0].numpy()
    # b = variables[1].numpy()


if __name__ == '__main__':
    model_name = "FM"
    main(model_name)