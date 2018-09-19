import os
import gzip
import time
import tempfile
import shutil
import requests
import tensorflow as tf


def del_file(path, retries=3, sleep=0.1):
    for i in range(retries):
        try:
            os.remove(path)
        except WindowsError:
            time.sleep(sleep)
        else:
            break


def download(directory, filename, url='http://yann.lecun.com/exdb/mnist/'):
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    temp_file, zipped_filepath = tempfile.mkstemp(suffix='.gz', text=False)
    print('Downloading %s to %s' % (url, zipped_filepath))
    os.write(temp_file, requests.get(url + filename + '.gz').content)
    os.close(temp_file)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
            tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def dataset(directory, images_file, labels_file):
    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)

    def decode_image(image):
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [28 * 28])
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)
        label = tf.reshape(label, [])
        return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def train(directory):
    return dataset(directory, 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')


def test(directory):
    return dataset(directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')


