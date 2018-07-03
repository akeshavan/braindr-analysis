import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import applications
from skimage.color import gray2rgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def using_GPU():
    return tf.test.gpu_device_name()


def convertToVGG(X):
    X_rgb = gray2rgb(X[:, :, :, 0])
    vgg_model = applications.VGG16(include_top=False, weights='imagenet',
                                   input_shape=(256, 256, 3))
    X_vgg = vgg_model.predict(X_rgb)
    return X_vgg


def get_split_indices(subjects_all, seed=0):
    np.random.seed(seed)
    idx = list(range(len(set(subjects_all))))
    subjects = np.asarray(list(set(subjects_all)))
    np.random.shuffle(idx)
    train_subs = idx[:int(0.8*subjects.shape[0])]
    test_subs = idx[int(0.8*subjects.shape[0]):int(0.9*subjects.shape[0])]
    val_subs = idx[int(0.9*subjects.shape[0]):]
    train = [i for i, val in enumerate(subjects_all) if
             val in subjects[train_subs]]
    np.random.shuffle(train)
    test = [i for i, val in enumerate(subjects_all) if
            val in subjects[test_subs]]
    np.random.shuffle(test)
    val = [i for i, val in enumerate(subjects_all) if
           val in subjects[val_subs]]
    np.random.shuffle(val)
    return train, test, val


def get_vgg_top_model(bottleneck_features_train, lr=1e-4):
    """A very simple model to train 1 final layer to combine the features
    from VGG16"""
    vggtop_model = Sequential()
    vggtop_model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))
    vggtop_model.add(Dense(256, activation='relu'))
    vggtop_model.add(Dropout(0.5))
    vggtop_model.add(Dense(1, activation='sigmoid'))

    vggtop_model.compile(optimizer=Adam(lr=lr),
                         loss='mean_squared_error',
                         metrics=['accuracy'])
    return vggtop_model


def run_model(X_vgg, y, img_labels, gold, out_path, i=0):
    # i is a random seed. This is set on the Tensorflow end on & the numpy end
    tf.set_random_seed(i)
    batch_size = 16
    epochs = 50
    filename = ("{out_path}/training%04d.csv" % i).format(out_path=out_path)
    filepath = ("{out_path}/best_model%04d.h5" % i).format(out_path=out_path)

    # get split indices, as a function of random seed
    train_idx, test_idx, val_idx = get_split_indices(img_labels.subject.values,
                                                     i)
    X_train, y_train = X_vgg[train_idx, :, :, :], y[train_idx]
    X_test, y_test = X_vgg[test_idx, :, :, :], y[test_idx]
    X_val, y_val = X_vgg[val_idx, :, :, :], y[val_idx]

    gold['in_test'] = gold.idx.isin(test_idx)
    gold_in = gold[gold.in_test]

    X_gold = X_vgg[gold_in['idx'].values]
    y_gold = gold_in['truth'].values

    vggtop_model = get_vgg_top_model(X_train, 1e-4)

    best_model = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)

    csv_logger = keras.callbacks.CSVLogger(filename, separator=',',
                                           append=False)

    callbacks = [best_model, csv_logger]

    vggtop_model.fit(X_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs, verbose=1,
                     callbacks=callbacks,
                     validation_data=(X_val, y_val))

    print(best_model.model.evaluate(X_gold, y_gold))
    y_gold_pred = best_model.model.predict(X_gold)
    gold_in['y_pred'] = y_gold_pred
    fpr_gold, tpr_gold, thresh_gold = roc_curve(y_gold, y_gold_pred)
    gold_auc = auc(fpr_gold, tpr_gold)
    print(gold_auc)
    return dict(model=best_model.model, fpr=fpr_gold, tpr=tpr_gold,
                threshold=thresh_gold, auc=gold_auc,
                training=pd.read_csv(filename),
                best_model=filepath, train_shape=X_train.shape,
                test_shape=X_test.shape, val_shape=X_val.shape,
                test_idx=test_idx, train_idx=train_idx, val_idx=val_idx,
                gold_test=gold_in, gold_idx=gold['idx'].values)
