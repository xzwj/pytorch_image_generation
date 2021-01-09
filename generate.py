"""Generates images"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

import utils
import models.nets as nets


parser = argparse.ArgumentParser()
parser.add_argument('--num', 
                    default=16, 
                    help="Number of images to generate")
parser.add_argument('--model', 
                    default='vae', 
                    choices=['vae', 'gan', 'wgan'],
                    help="Choose model (vae, gan, wgan)")
parser.add_argument('--model_dir', 
                    default='experiments/mnist_vae',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', 
                    default='best',
                    choices=['best', 'last'],
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")


def generate(num, model, params, model_dir):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    # create result_dir if it does not exists
    result_dir = os.path.join(model_dir, 'results') 
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with torch.no_grad():
        # sample z from N(0,1)
        z_batch = Variable(torch.randn(num, params.latent_dim))

        # move to GPU if available
        if params.cuda:
            z_batch = z_batch.cuda(non_blocking=True)

        # generate images
        gen_batch = model.decode(z_batch)
        gen_batch = gen_batch.view(num, params.input_channel, params.input_h, params.input_w)

        save_image(gen_batch.cpu(), os.path.join(result_dir, 'generation.png'), nrow=num)




if __name__ == '__main__':
    """
    Generate images using sampled z
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(138)
    if params.cuda:
        torch.cuda.manual_seed(138)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'generate.log'))

    # Define model
    if args.model == 'vae':
        model = nets.VAE(params).cuda() if params.cuda else nets.VAE(params)
    elif args.model == 'gan':
        model = nets.GAN(params).cuda() if params.cuda else nets.GAN(params)
    else:
        model = nets.WGAN(params).cuda() if params.cuda else nets.WGAN(params)

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Generate
    generate(args.num, model, params, args.model_dir)


    