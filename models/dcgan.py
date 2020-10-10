import torch.nn as nn
from torch import optim
from utils.util import *

from layers.discriminator import Discriminator
from layers.generator import Generator
from utils.const import *

logger = get_logger(__name__)


class DCGAN(object):
    def __init__(self, args):
        logger.info("DCGAN model initalization.")
        self.G = Generator()
        self.D = Discriminator(sigmoid=True)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=LEARNING_RATE_D, betas=(BEATA_1, BEATA_2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=LEARNING_RATE_G, betas=(BEATA_1, BEATA_2))
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.loss = nn.BCELoss()

    def train(self, batches, args: Arguments):
        start_time = t.time()
        generator_iter = 0
        for epoch in range(self.epochs):
            epoch_start_time = t.time()
            data = get_batches_data(args, batches)
            logger.info("Start epoch {}".format(epoch))
            for i, (real_images, _) in enumerate(data):
                if i == data.dataset.__len__() // self.batch_size:
                    logger.info("wrong number of images")
                    break
                self.D.zero_grad()
                real_labels = torch.ones(self.batch_size)
                fake_labels = torch.zeros(self.batch_size)
                outputs = self.D(real_images).view(-1)
                d_loss_real = self.loss(outputs, real_labels)
                d_loss_real.backward()
                noise = torch.randn(self.batch_size, Z_VECTOR_SIZE, 1, 1)
                fake_images = self.G(noise)
                outputs = self.D(fake_images).view(-1)
                d_loss_fake = self.loss(outputs, fake_labels)
                d_loss_fake.backward()
                d_loss = d_loss_real + d_loss_fake
                self.d_optimizer.step()
                fake_images = self.G(noise)
                self.G.zero_grad()
                outputs = self.D(fake_images).view(-1)
                g_loss = self.loss(outputs, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1
                if generator_iter % SAVE_IMAGE_ITERATIONS == 0:
                    logger.info('Epoch: {}, Iteration: {}, g_loss: {}'.format(epoch, generator_iter, g_loss))
                    logger.info('Epoch: {}, Iteration: {}, d_loss_fake: {}'.format(epoch, generator_iter, d_loss_fake))
                    logger.info('Epoch: {}, Iteration: {}, d_loss_real: {}'.format(epoch, generator_iter, d_loss_real))
                    logger.info('Epoch: {}, Iteration: {}, d_loss: {}'.format(epoch, generator_iter, d_loss))
                save_iteration(self.G, self.D, generator_iter, args.number_of_rows, args.output_images, start_time)

            end_time = t.time()
            logger.info('Finished epoch {} in {}'.format(epoch, (end_time - epoch_start_time)))

        end_time = t.time()
        logger.info('Time of training-{}'.format((end_time - start_time)))
        save_model(self.G.state_dict(), self.D.state_dict())
