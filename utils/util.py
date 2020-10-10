import logging
import os
import typing
import time as t
import torch
from torchvision.utils import make_grid, save_image
from dataset_preparation.cifar import CIFAR10
import torch.utils.data as data_utils

from layers.discriminator import Discriminator
from utils.const import *
from layers.generator import Generator

logging.getLogger('botocore').setLevel(logging.CRITICAL)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    if len(logger.handlers) == 0:
        logger.addHandler(ch)
    logger.propagate = 0
    return logger


logger = get_logger(__name__)


def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images


def save_model(generator, discriminator):
    torch.save(generator, './generator.pkl')
    torch.save(discriminator, './discriminator.pkl')
    logger.info('Models save to ./generator.pkl & ./discriminator.pkl ')


def load_batches() -> typing.List:
    batches = []
    for file_name in os.listdir(os.path.join(CIFAR_ROOT, BATCH_FOLDER)):
        batches.append(file_name)
    return batches


def save_iteration(generator: Generator, discriminator: Discriminator, g_iter, number_of_rows, output_images,
                   start_time):
    if g_iter % SAVE_IMAGE_ITERATIONS == 0:
        save_model(generator.state_dict(), discriminator.state_dict())
        if not os.path.exists('training_result_images/'):
            os.makedirs('training_result_images/')
        generate_fake_images(g_iter, generator, number_of_rows, output_images)

        time = t.time() - start_time
        logger.info("Generator iter: {}".format(g_iter))
        logger.info("Time {}".format(time))


def generate_fake_images(g_iter, generator, number_of_rows, output_images):
    noise = torch.randn(RANDOM_IMAGES, Z_VECTOR_SIZE, 1, 1)
    samples = generator(noise)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.data.cpu()[:output_images]
    grid = make_grid(samples, number_of_rows)
    save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)),
               nrow=number_of_rows)


def get_batches_data(args, batches):
    train_dataset = CIFAR10(batches, args.image_size, train=True)
    logger.info("finished initializing dataset")
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader


def load_model(model):
    logger.info('Start load model')
    generator = os.path.join('generator.pkl')
    model.G.load_state_dict(torch.load(generator))
    logger.info('Finished load model')


class Arguments(typing.NamedTuple):
    model: str
    epochs: int
    train: bool
    create_batches: bool
    batch_size: int
    generator_iters: int
    output_images: int
    number_of_rows: int
    image_size: int
