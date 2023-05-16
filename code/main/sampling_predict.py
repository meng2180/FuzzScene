import sys, random
import os

sys.path.append("..")
from configs import bcolors
from utils import *

import multiprocessing


def preprocess(path, target_size):
    return preprocess_image(path, target_size)[0]


def data_generator(xs, ys, target_size, batch_size=64):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess(x, target_size) for x in paths]
            gen_state = 0
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess(x, target_size) for x in paths]
            gen_state += batch_size
        yield np.array(X), np.array(y)


def load_carla_test_data(path='', batch_size=32, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()

    with open(path + 'label_test.csv', 'r') as f:
        rows = len(f.readlines()) - 1
        f.seek(0)
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + 'center/' + line.split(',')[0])
            ys.append(1)

    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    train_generator = data_generator(train_xs, train_ys,
                                     target_size=shape,
                                     batch_size=batch_size)

    print(bcolors.OKBLUE + 'finished loading data, running time: {} seconds'.format(
        time.time() - start_load_time) + bcolors.ENDC)
    return train_generator, len(train_xs)


def predict(model_name, seed_number, q):
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    batch_size = 8
    image_shape = (100, 100)
    test_dataset_path = '../scenario_runner-0.9.13/_out/'

    model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(100, 100, 3),
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
    for layer in model.layers:
        layer.trainable = False

    model.summary()

    x = tf.keras.layers.Flatten()(model.output)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
    head_model = tf.keras.Model(inputs=model.input, outputs=predictions)

    s = '../sampling_vgg/vgg_' + model_name + '_' + seed_number + '.h5'
    head_model.load_weights(s)
    head_model.compile(optimizer='adam',
                       loss=tf.keras.losses.sparse_categorical_crossentropy,
                       metrics=['accuracy'])

    # --------------------------------------Evaluation---------------------------------------- #
    K.set_learning_phase(0)
    test_generator, samples_per_epoch = load_carla_test_data(path=test_dataset_path, batch_size=batch_size,
                                                             shape=image_shape)
    print('test samples: ', samples_per_epoch)
    loss = head_model.evaluate(test_generator, steps=math.ceil(samples_per_epoch * 1. / batch_size), verbose=1)

    prediction = head_model.predict(test_generator, steps=math.ceil(samples_per_epoch * 1. / batch_size), verbose=1)
    sum_pre = 0
    print(prediction)
    for i in range(len(prediction)):
        if prediction[i][1] > prediction[i][0]:
            sum_pre = sum_pre + 1
    q.put(sum_pre)
    return None


def prenum(model_name, seed_number):
    with open('../scenario_runner-0.9.13/_out/label_test.csv', 'r') as f:
        rows = len(f.readlines()) - 1

        with open('sample_num_vgg.csv', 'a+', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            csv_writer.writerow([timestr, model_name, seed_number, rows])
        if (rows == 0):
            return 0

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=predict, args=(model_name, seed_number, q))
    p.start()
    p.join()
    number = q.get()
    return number
