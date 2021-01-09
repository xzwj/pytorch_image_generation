"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

import utils
import models.nets as nets
import models.data_loaders as data_loaders


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
                    default='mnist', 
                    choices=['cifar10', 'mnist'],
                    help="Choose dataset (cifar10 or mnist)")
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


def evaluate(model, loss_fn, dataloader, params, model_dir, epoch=-1):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyperparameters
        epoch: (int) epoch=-1 means it is evaluation for testing data; 
                epoch>=0 means it is evaluation for training data
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # create result_dir if it does not exists
    result_dir = os.path.join(model_dir, 'results') 
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with torch.no_grad():
        # compute metrics over the dataset
        for ii, (data_batch, _) in enumerate(dataloader):

            # move to GPU if available
            if params.cuda:
                data_batch = data_batch.cuda(non_blocking=True)
            # fetch the next evaluation batch
            data_batch = Variable(data_batch)

            # compute model output
            recon_batch, mu, logvar = model(data_batch)
            loss = loss_fn(recon_batch, data_batch, mu, logvar)

            # # extract data from torch Variable, move to cpu, convert to numpy arrays
            # output_batch = output_batch.data.cpu().numpy()
            # labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

            if ii == 0:
                n = min(data_batch.size(0), 16)
                comparison = torch.cat([data_batch[:n], recon_batch.view(params.batch_size, \
                                        params.input_channel, params.input_h, params.input_w)[:n]])
                save_image(comparison.cpu(),
                            os.path.join(result_dir, 'reconstruction_' + str(epoch) + '.png'), nrow=n)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.sum([x[metric] for x in summ]) / len(dataloader.dataset) \
                    for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean




if __name__ == '__main__':
    """
    Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(138)
    if params.cuda:
        torch.cuda.manual_seed(138)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    if args.dataset == 'cifar10':
        dataloaders = data_loaders.cifar10_dataloader(params.batch_size)
    else:
        dataloaders = data_loaders.mnist_dataloader(params.batch_size)
    val_dl = dataloaders['test']

    logging.info("- done.")

    # Define model
    if args.model == 'vae':
        model = nets.VAE(params).cuda() if params.cuda else nets.VAE(params)
    elif args.model == 'gan':
        model = nets.GAN(params).cuda() if params.cuda else nets.GAN(params)
    else:
        model = nets.WGAN(params).cuda() if params.cuda else nets.WGAN(params)
    
    # fetch loss function and metrics
    loss_fn = nets.vae_loss_fn
    # metrics = nets.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, params, args.model_dir, -1)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)


    