import os
import shutil
import argparse
import csv
import cv2
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Cropping2D
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
import keras.backend.tensorflow_backend as ktf

def initialize(args):
    path = output_path('', args)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    os.system('sh versions.sh > {}'.format(output_path('VERSIONS', args)))

def get_session(gpu_fraction=1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def get_samples(data_dir):
    samples = []
    csv_path = '{}/driving_log.csv'.format(data_dir)
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile, )
        for line in reader:
            samples.append(line)
    return samples

def get_image(data_dir, path):
    filename = path.split('/')[-1]
    filepath = '{}/IMG/{}'.format(data_dir, filename)
    return cv2.imread(filepath)

def output_path(filename, args):
    return 'outputs/{}/{}'.format(args.output_folder, filename)

def generator(data_dir, samples, batch_size=32, correction=0.25):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = get_image(data_dir, batch_sample[0])
                left_image = get_image(data_dir, batch_sample[1])
                right_image = get_image(data_dir, batch_sample[2])
                
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])
                
                def flip(image, angle):
                    image_flipped = np.fliplr(image)
                    angle_flipped = -angle
                    images.append(image_flipped)
                    angles.append(angle_flipped)

                flip(center_image, center_angle)
                flip(left_image, left_angle)
                flip(right_image, right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def generate_model(keep_prob=0.25):
    model = Sequential()
    model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/256 - 0.5))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu"))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation="relu"))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation="relu"))
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def summarize(model, args):
    model.summary()

    s = ''
    for arg in vars(args):
        s += '{} : {}\n'.format(arg, getattr(args, arg))

    with open(output_path(args.summary_filename, args), 'w') as fh:
        fh.write(s + '\n')
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

def main():
    parser = argparse.ArgumentParser(description='Deep learning model generator based on input data')
    parser.add_argument(
        '--data-dir',
        help='Data directory path (default: data)',
        dest='data_dir',  
        type=str,  
        default='data')
    parser.add_argument(
        '--test-size',
        help='Test size (default: 0.2)',
        dest='test_size', 
        type=float,
        default=0.2)
    parser.add_argument(
        '--keep-prob',
        help='Drop out probability (default: 0.25)',
        dest='keep_prob', 
        type=float,
        default=0.25)
    parser.add_argument(
        '--epochs',
        help='Number of epochs (default: 7)',
        dest='epochs',  
        type=int,  
        default=7)
    parser.add_argument(
        '--batch-size',
        help='Batch size (default: 32)',
        dest='batch_size',
        type=int,  
        default=32)
    parser.add_argument(
        '--correction',
        help='Correction of left/right steering angle (default: 0.25)',
        dest='correction',
        type=float,
        default=0.25)
    parser.add_argument(
        '--output-folder',
        help='Name of the output folder (default: output)',
        dest='output_folder',
        type=str,  
        default='output')
    parser.add_argument(
        '--summary-filename',
        help='Name of the summary file (default: SUMMARY)',
        dest='summary_filename',
        type=str,
        default='SUMMARY')
    args = parser.parse_args()

    initialize(args)
    ktf.set_session(get_session())

    samples = get_samples(args.data_dir);
    train_samples, validation_samples=train_test_split(samples, test_size=args.test_size)

    train_generator = generator(args.data_dir, train_samples, batch_size=args.batch_size, correction=args.correction)
    validation_generator = generator(args.data_dir, validation_samples, batch_size=args.batch_size, correction=args.correction)

    model = generate_model(args.keep_prob)
    summarize(model, args)

    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_samples)*3*2,
                        validation_data=validation_generator,
                        validation_steps=len(validation_samples),
                        epochs=args.epochs)
    model.save(output_path('model.h5', args))

if __name__ == "__main__":
    main()
