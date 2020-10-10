import os
import shutil
import typing
import pickle
import numpy
from random import shuffle

from PIL import Image
from utils.util import get_logger
from utils.const import *

logger = get_logger(__name__)


def create_raw_dataset(image_size: int) -> typing.Set:
    normalized_images_path = os.path.join(DATASET_PATH, "normalized_images")
    if not os.path.exists(normalized_images_path):
        os.mkdir(normalized_images_path)
    try:
        normalized_train_path = os.path.join(normalized_images_path, NORMALIZED_TRAIN_FOLDER)
        if not os.path.exists(normalized_train_path):
            os.mkdir(normalized_train_path)

        normalize_images(os.path.join(DATASET_PATH, TRAIN_FOLDER), normalized_train_path, image_size)
        normalize_folders = [[normalized_train_path, 'train']]
        batches = []
        for name in normalize_folders:
            batches = batches + (create_batches_files(name, image_size))
        logger.info("batch creation is done")
        return set(batches)
    finally:
        shutil.rmtree(normalized_images_path, ignore_errors=True)


def normalize_images(source_path: str, target_path: str, image_size: int):
    for folder_name in os.listdir(source_path):
        if not os.path.exists(os.path.join(target_path, folder_name)):
            os.mkdir(os.path.join(target_path, folder_name))
        for file_name in os.listdir(os.path.join(source_path, folder_name)):
            try:
                image = Image.open(os.path.join(source_path, folder_name, file_name))
                new_image = image.resize((image_size, image_size), Image.ANTIALIAS)
                new_image.save(os.path.join(target_path, folder_name, file_name), quality=100)
            except OSError as e:
                logger.info("This file: {} could not be converted {}".format(file_name, e))
            except ValueError as e:
                logger.info("This file: {} could not be converted {}".format(file_name, e))


def create_batches_files(name: typing.List, image_size: int) -> typing.List:
    file_list = []
    for dirname in os.listdir(name[0]):
        path = os.path.join(name[0], dirname)
        for filename in os.listdir(path):
            file_list.append(os.path.join(name[0], dirname, filename))
    logger.info("start create batches")
    shuffle(file_list)
    batches = []
    img_values = []
    labels = []
    entry = {}
    number_of_picture = 0
    counter = 1
    for filename in file_list:
        label = int(filename.split('/')[len(filename.split('/')) - 2])
        image = Image.open(filename)
        pixel_values = list(image.getdata())
        pixel_values = numpy.array(pixel_values)
        tmp_img_values = []
        if len(pixel_values[0]) == 3 and len(pixel_values) == image_size * image_size:
            for col in range(0, len(pixel_values[0])):
                for row in range(0, len(pixel_values)):
                    tmp_img_values.append(pixel_values[row, col])
            img_values.append(tmp_img_values)
            labels.append(label)
            number_of_picture += 1
            if number_of_picture % 2000 == 0:
                logger.info("finished {} pictures".format(number_of_picture))
            if number_of_picture % MAX_BATCH_IMAGES == 0:
                batch = 'data_batch_' + str(counter)
                save_new_batch(batch, entry, img_values, labels)
                batches.append(batch)
                entry = {}
                img_values = []
                labels = []
                logger.info("finished {} batches".format(counter))
                counter += 1

    batch = 'data_batch_' + str(counter)
    save_new_batch(batch, entry, img_values, labels)
    batches.append(batch)
    logger.info("finished create batches")
    return batches


def save_new_batch(batch, entry, img_values, labels):
    entry['data'] = numpy.array(img_values, dtype=numpy.uint8)
    entry['labels'] = labels
    new_batch = os.path.join(DATASET_PATH, CIFAR_FOLDER, batch)
    with open(new_batch, 'wb') as f:
        pickle.dump(entry, f)
