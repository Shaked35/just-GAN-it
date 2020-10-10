import argparse
from dataset_preparation.image_conversion import create_raw_dataset
from models.dcgan import DCGAN
from models.wgan import WGAN
from utils.util import *

logger = get_logger(__name__)


def main(args: Arguments):
    logger.info("start GAN process")
    if args.model == "DCGAN":
        model = DCGAN(args)
    elif args.model == "WGAN":
        model = WGAN(args)
    else:
        raise Exception("There is no model")
    if args.train:
        if args.create_batches:
            batches = create_raw_dataset(args.image_size)
        else:
            batches = load_batches()
        logger.info(batches)
        logger.info("start training model")
        model.train(batches, args)
        logger.info("finished training model")
        logger.info("finished GAN process")
    else:
        load_model(model)
        generate_fake_images("generate_new_image", model.G, args.number_of_rows, args.output_images)


def _arg_parser() -> Arguments:
    parser = argparse.ArgumentParser(description="feature selection optimization")
    parser.add_argument("-m", help="model", type=str, required=True)
    parser.add_argument("-e", help="epochs", type=int, required=False)
    parser.add_argument("-bs", help="batch_size", type=int, required=True)
    parser.add_argument("-gi", help="generator_iters", type=int, required=False)
    parser.add_argument("-oi", help="output_images", type=int, required=True)
    parser.add_argument("-nor", help="number_of_rows", type=int, required=True)
    parser.add_argument("-ims", help="image_size", type=int, required=True)
    parser.add_argument("-cb", help="create_batches", type=str, required=False)
    parser.add_argument("-t", help="train", type=str, required=False)
    parsed = parser.parse_args()

    if parsed.cb == 'False':
        create_batches = False
    else:
        create_batches = True
    if parsed.t == 'False':
        train = False
    else:
        train = True
    return Arguments(model=parsed.m, epochs=parsed.e, batch_size=parsed.bs, generator_iters=parsed.gi,
                     output_images=parsed.oi,
                     number_of_rows=parsed.nor, image_size=parsed.ims, create_batches=create_batches, train=train)


if __name__ == "__main__":
    main(_arg_parser())
