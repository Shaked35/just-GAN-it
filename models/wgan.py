import matplotlib.pyplot as plt
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from utils.util import *
from layers.discriminator import Discriminator
from layers.generator import Generator
plt.switch_backend('agg')
logger = get_logger(__name__)


class WGAN(object):
    def __init__(self, args):
        logger.info("initialize WGAN model")
        self.image_size = args.image_size
        self.G = Generator()
        self.D = Discriminator(sigmoid=False)
        self.batch_size = args.batch_size
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=LEARNING_RATE_D, betas=(BEATA_1, BEATA_2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=LEARNING_RATE_G, betas=(BEATA_1, BEATA_2))
        self.generator_iters = args.generator_iters

    def train(self, batches, args: Arguments):
        start_time = t.time()
        data_loader = get_batches_data(args, batches)
        data = get_infinite_batches(data_loader)
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        for g_iter in range(self.generator_iters):
            for p in self.D.parameters():
                p.requires_grad = True

            for d_iter in range(CRITIC_ITER):
                d_loss_fake, d_loss_real, fake_images, images = self.get_real_and_fake_images(data, mone, one)
                if images.size()[0] != self.batch_size:
                    logger.info(
                        'not enough images in this batch. images {}, batch {}'.format(images.size()[0],
                                                                                      self.batch_size))
                    continue
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()
                self.d_optimizer.step()
                logger.info(
                    'Discriminator iteration: {}/{}, loss_fake: {}, loss_real: {}'.format(d_iter, CRITIC_ITER,
                                                                                          d_loss_fake, d_loss_real))

            for p in self.D.parameters():
                p.requires_grad = False
            self.G.zero_grad()
            noise = torch.randn(self.batch_size, Z_VECTOR_SIZE, 1, 1)
            fake_images = self.G(noise)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            self.g_optimizer.step()
            logger.info('Generator iteration: {}/{}, g_loss: {}'.format(g_iter, self.generator_iters, g_loss))
            save_iteration(self.G, self.D, g_iter, args.number_of_rows, args.output_images, start_time)

        end_time = t.time()
        logger.info('Time of training-{}'.format((end_time - start_time)))

    def get_real_and_fake_images(self, data, mone, one):
        self.D.zero_grad()
        images = data.__next__()
        d_loss_real = self.D(images)
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(mone)
        noise = torch.randn(self.batch_size, Z_VECTOR_SIZE, 1, 1)
        fake_images = self.G(noise)
        d_loss_fake = self.D(fake_images)
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(one)
        return d_loss_fake, d_loss_real, fake_images, images

    def calculate_gradient_penalty(self, real_images, fake_images):
        weight = torch.FloatTensor(self.batch_size, 1, 1, 1).uniform_(0, 1)
        weight = weight.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        interpolated = weight * real_images + ((1 - weight) * fake_images)
        interpolated = Variable(interpolated, requires_grad=True)
        prob_interpolated = self.D(interpolated)
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA_FACTOR

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images
