from configs import bcolors
from utils import *

from imageio import imread
from imageio import imsave
from PIL import Image
from scipy import ndimage


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


def load_carla_train_data(path='', batch_size=32, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()

    with open(path + 'label_train.csv', 'r') as f:
        rows = len(f.readlines()) - 1
        f.seek(0)
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + 'center/' + line.split(',')[0])
            ys.append(float(line.split(',')[2]))

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
            ys.append(float(line.split(',')[2]))

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


def carla_load_steering_data(steering_log):
    df_steer = pd.read_csv(steering_log, usecols=['frame_id', 'steering_angle_change'], index_col=False)

    angle = np.zeros((df_steer.shape[0], 1))
    time = np.zeros((df_steer.shape[0], 1), dtype=np.int32)

    angle[:, 0] = df_steer['steering_angle_change'].values
    frame_arr = []
    for frame in df_steer['frame_id']:
        frame_arr.append(int(frame[:-4]))
    time[:, 0] = frame_arr

    data = np.append(time, angle, axis=1)

    return data


def carla_load_frame_id(data):
    frame_id = []
    for i in range(0, data.shape[0]):
        frame_id.append(int(data[i, 0]))
    return frame_id


def normalize_input(x):
    return x / 255.


def exact_output(y):
    return y


def carla_read_steerings(steering_log):
    steerings = defaultdict(list)
    with open(steering_log) as f:
        for line in f.readlines()[1:]:
            fields = line.split(",")
            timestamp, angle = int(str(fields[0])[:-4]), float(fields[2])
            steerings[timestamp].append(angle)
    return steerings


def carla_read_images(image_folder, id, image_size):
    prefix = path.join(image_folder, 'center')
    img_path = path.join(prefix, '%08d.png' % id)
    imgs = []

    img = imread(img_path, pilmode='RGB')

    # Cropping
    crop_img = img[200:, :]

    # Resizing
    img = np.array(Image.fromarray(crop_img).resize(image_size))

    imgs.append(img)

    if len(imgs) < 1:
        print('Error no image at timestamp')
        print(id)

    img_block = np.stack(imgs, axis=0)

    if K.image_data_format() == 'channels_first':
        img_block = np.transpose(img_block, axes=(0, 3, 1, 2))

    return img_block


def carla_read_images_augment(image_folder, id, image_size):
    prefix = path.join(image_folder, 'center')
    img_path = path.join(prefix, '%08d.png' % id)
    imgs = []

    img = imread(img_path, pilmode='RGB')

    # Flip image
    img = np.fliplr(img)

    # Cropping
    crop_img = img[200:, :]

    # Resizing
    # img = imresize(crop_img, output_shape=image_size)
    img = np.array(Image.fromarray(crop_img).resize(image_size))

    # Rotate randomly by small amount (not a viewpoint transform)
    rotate = random.uniform(-1, 1)

    img = ndimage.rotate(img, rotate, reshape=False)

    imgs.append(img)

    if len(imgs) < 1:
        print('Error no image at timestamp')
        print(id)

    img_block = np.stack(imgs, axis=0)

    if K.image_data_format() == 'channels_first':
        img_block = np.transpose(img_block, axes=(0, 3, 1, 2))

    return img_block


def camera_adjust(angle, speed, camera):
    # Left camera -20 inches, right camera +20 inches (x-direction)
    # Steering should be correction + current steering for center camera

    # Chose a constant speed
    speed = 10.0  # Speed

    # Reaction time - Time to return to center
    # The literature seems to prefer 2.0s (probably really depends on speed)
    if speed < 1.0:
        reaction_time = 0
        angle = angle
    else:
        reaction_time = 2.0  # Seconds

        # Trig to find angle to steer to get to center of lane in 2s
        opposite = 20.0  # inches
        adjacent = speed * reaction_time * 12.0  # inches (ft/s)*s*(12 in/ft) = inches (y-direction)
        angle_adj = np.arctan(float(opposite) / adjacent)  # radians

        # Adjust based on camera being used and steering angle for center camera
        if camera == 'left':
            angle_adj = -angle_adj
        angle = angle_adj + angle

    return angle


def carla_data_generator(frame_id, steering_log, image_folder, unique_list, gen_type='train',
                         batch_size=32, image_size=(128, 128), shuffle=True,
                         preprocess_input=normalize_input, preprocess_output=exact_output):
    # Read all steering angles , get <frame_id, steering> map
    # -----------------------------------------------------------------------------
    steerings = carla_read_steerings(steering_log)

    # Data debug info
    # -----------------------------------------------------------------------------
    start = min(unique_list)
    end = max(unique_list)

    i = 0
    x_buffer, y_buffer, buffer_size = [], [], 0
    while True:
        if i > end:
            i = start

        coin = random.randint(1, 2)

        if steerings[i] and i in frame_id:
            if gen_type == 'train':
                if coin == 1:
                    image = carla_read_images(image_folder, i, image_size)
                else:
                    image = carla_read_images_augment(image_folder, i, image_size)
            else:
                image = carla_read_images(image_folder, i, image_size)

            # Mean angle with a timestamp
            angle = np.repeat([steerings[i][0]], image.shape[0])

            # Adjust steering angle for horizontal flipping
            if gen_type == 'train' and coin == 2:
                angle = -angle

            # Adjust the steerings of the offcenter cameras
            x_buffer.append(image)
            y_buffer.append(angle)
            buffer_size += image.shape[0]

            if buffer_size >= batch_size:
                indx = list(range(buffer_size))
                if gen_type == 'train':
                    np.random.shuffle(indx)
                x = np.concatenate(x_buffer, axis=0)[indx[:batch_size], ...]
                y = np.concatenate(y_buffer, axis=0)[indx[:batch_size], ...]

                x_buffer, y_buffer, buffer_size = [], [], 0
                yield preprocess_input(x.astype(np.float32)), preprocess_output(y)

        if shuffle:
            i = int(random.choice(unique_list))
        else:
            i += 1
            while i not in unique_list:
                i += 1
                if i > end:
                    i = start
