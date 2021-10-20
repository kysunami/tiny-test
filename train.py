"""Main entrance for train/eval with/without KD on TinyImagenet"""

import argparse
import logging
import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.data_loader as data_loader
from evaluate import evaluate, evaluate_kd

import model.lenet5 as lenet5
import model.lenet5m as lenet5m
import model.lenet5n as lenet5n
import model.lenet5n6 as lenet5n6
import model.lenet5n10 as lenet5n10
import model.lenet5n1 as lenet5n1
import model.net as net
import model.preresnet as preresnet
import model.resnet as resnet
import model.pyramidnet as pyramidnet
from model.utils import (get_model_params, BlockDecoder)
import torchvision.models as models

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data-cifar10', help="Directory for the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch) #memory leak

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()
                #print(metric.item() for metric in metrics)

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data.item()#[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data.item())#[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    #metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    #summary()


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    # learning rate schedulers for different models:
    if params.model_version == "resnet18":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "resnet50":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "resnet110":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "fresnet110":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "preresnet110":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "pyramidnet":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "vibnet":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "vgg19":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5m":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5n":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5n6":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5n10":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5n1":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)


    for epoch in range(params.num_epochs):
     
        scheduler.step()
     
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)        

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


# Helper function: get [batch_idx, teacher_outputs] list by running teacher model once
def fetch_teacher_outputs(teacher_model, dataloader, params):
    # set teacher_model to evaluation mode
    teacher_model.eval()
    teacher_outputs = []
    for i, (data_batch, labels_batch) in enumerate(dataloader):
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), \
                                        labels_batch.cuda(async=True)
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        output_teacher_batch = teacher_model(data_batch).data.cpu().numpy()
        teacher_outputs.append(output_teacher_batch)

    return teacher_outputs


# Defining train_kd & train_and_evaluate_kd functions
def train_kd(model, teacher_outputs, optimizer, loss_fn_kd, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()
    # teacher_model.eval()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), \
                                            labels_batch.cuda(async=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output, fetch teacher output, and compute KD loss
            output_batch = model(train_batch)

            # get one batch output from teacher_outputs list
            output_teacher_batch = torch.from_numpy(teacher_outputs[i])
            if params.cuda:
                output_teacher_batch = output_teacher_batch.cuda(async=True)
            output_teacher_batch = Variable(output_teacher_batch, requires_grad=False)

            loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data.item()#[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data.item())#[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                       loss_fn_kd, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - file to restore (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    
    # Tensorboard logger setup
    # board_logger = utils.Board_Logger(os.path.join(model_dir, 'board_logs'))

    # fetch teacher outputs using teacher_model under eval() mode
    loading_start = time.time()
    teacher_model.eval()
    teacher_outputs = fetch_teacher_outputs(teacher_model, train_dataloader, params)
    elapsed_time = math.ceil(time.time() - loading_start)
    logging.info("- Finished computing teacher outputs after {} secs..".format(elapsed_time))

    # learning rate schedulers for different models:
    if params.model_version == "resnet18_distill":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1) 
    elif params.model_version == "resnet50_distill":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1) 
    elif params.model_version == "resnet110_distill":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1) 
    elif params.model_version == "fresnet110_distill":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1) 
    elif params.model_version == "preresnet_distill":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1) 
    elif params.model_version == "pyramidnet_distill":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1) 
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn_distill": 
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5_distill":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5m_distill":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5n_distill":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5n6_distill":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5n10_distill":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version == "lenet5n1_distill":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)

    for epoch in range(params.num_epochs):

        scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_kd(model, teacher_outputs, optimizer, loss_fn_kd, train_dataloader,
                 metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate_kd(model, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


        # #============ TensorBoard logging: uncomment below to turn in on ============#
        # # (1) Log the scalar values
        # info = {
        #     'val accuracy': val_acc
        # }

        # for tag, value in info.items():
        #     board_logger.scalar_summary(tag, value, epoch+1)

        # # (2) Log values and gradients of the parameters (histogram)
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     board_logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
        #     # board_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    random.seed(230)
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders, considering full-set vs. sub-set scenarios
    if params.subset_percent < 1.0:
        train_dl = data_loader.fetch_subset_dataloader('train', params)
    else:
        train_dl = data_loader.fetch_dataloader('train', params)
    
    dev_dl = data_loader.fetch_dataloader('dev', params)

    logging.info("- done.")

    if "distill" in params.model_version:

        # train a 5-layer CNN or a 18-layer ResNet with knowledge distillation
        if params.model_version == "cnn_distill":
            model = net.Net(params).cuda() if params.cuda else net.Net(params)
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = net.metrics
            student_checkpoint = 'experiments/cnn_distill/'#karuna 
        
        elif params.model_version == 'lenet5_distill':
            model = lenet5.LeNet5(channels=3, class_count=200, act='relu').cuda() if params.cuda else lenet5.LeNet5(channels=3, class_count=200, act='relu')
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = lenet5.metrics
            student_checkpoint = 'experiments/lenet5_distill/'#karuna
        
        elif params.model_version == 'lenet5m_distill':
            model = lenet5m.LeNet5m().cuda() if params.cuda else lenet5m.LeNet5m()
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = lenet5m.metrics
            student_checkpoint = 'experiments/lenet5m_distill/'#karuna
        
        elif params.model_version == 'lenet5n_distill':
            model = lenet5n.LeNet5n(channels=3, class_count=200, act='relu').cuda() if params.cuda else lenet5n.LeNet5n(channels=3, class_count=200, act='relu')
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = lenet5n.metrics
            student_checkpoint = 'experiments/lenet5n_distill/'#karuna
        
        elif params.model_version == 'lenet5n6_distill':
            model = lenet5n6.LeNet5n6(channels=3, class_count=200, act='relu').cuda() if params.cuda else lenet5n6.LeNet5n6(channels=3, class_count=200, act='relu')
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = lenet5n6.metrics
            student_checkpoint = 'experiments/lenet5n6_distill/'#karuna
        
        elif params.model_version == 'lenet5n10_distill':
            model = lenet5n10.LeNet5n10(channels=3, class_count=200, act='relu').cuda() if params.cuda else lenet5n10.LeNet5n10(channels=3, class_count=200, act='relu')
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = lenet5n10.metrics
            student_checkpoint = 'experiments/lenet5n10_distill/'#karuna
        
        elif params.model_version == 'lenet5n1_distill':
            model = lenet5n1.LeNet5n1(channels=3, class_count=200, act='relu').cuda() if params.cuda else lenet5n1.LeNet5n1(channels=3, class_count=200, act='relu')
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = lenet5n1.metrics
            student_checkpoint = 'experiments/lenet5n1_distill/'#karuna 

        elif params.model_version == 'resnet18_distill':
            model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = resnet.metrics
            student_checkpoint = 'experiments/resnet18_distill/'#karuna

        elif params.model_version == 'resnet50_distill':
            model = resnet.ResNet50().cuda() if params.cuda else resnet.ResNet50()
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = resnet.metrics
            student_checkpoint = 'experiments/resnet50_distill/'#karuna

        elif params.model_version == 'resnet110_distill':
            model = resnet.ResNet110().cuda() if params.cuda else resnet.ResNet110()
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = resnet.metrics
            student_checkpoint = 'experiments/resnet110_distill/'#karuna

        elif params.model_version == 'fresnet110_distill':
            model = models.resnet18(num_classes=200).cuda() if params.cuda else models.resnet18(num_classes=200)
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = resnet.metrics
            student_checkpoint = 'experiments/fresnet110_distill/'#karuna

        elif params.model_version == 'preresnet_distill':
            model = preresnet.PreResNet(depth=110, num_classes=200).cuda() if params.cuda else preresnet.PreResNet(depth=110, num_classes=200)
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = preresnet.metrics
            student_checkpoint = 'experiments/preresnet_distill/'#karuna
        
        elif params.model_version == 'pyramidnet_distill':
            model = pyramidnet.PyramidNet(dataset='cifar10', depth=272, alpha=200, num_classes=200, bottleneck=True).cuda() if params.cuda else pyramidnet.PyramidNet(dataset='cifar10', depth=272, alpha=200, num_classes=200, bottleneck=True)
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = pyramidnet.metrics
            student_checkpoint = 'experiments/pyramidnet_distill/'#karuna

        if params.teacher == "resnet18":
            teacher_model = resnet.ResNet18()
            teacher_checkpoint = 'experiments/base_resnet18/best.pth.tar'
            teacher_model = teacher_model.cuda() if params.cuda else teacher_model
            student_checkpoint = student_checkpoint + 'resnet18_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)

        elif params.teacher == "resnet50":
            teacher_model = resnet.ResNet50()
            teacher_checkpoint = 'experiments/base_resnet50/best.pth.tar'
            teacher_model = teacher_model.cuda() if params.cuda else teacher_model
            student_checkpoint = student_checkpoint + 'resnet50_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)

        elif params.teacher == "resnet110":
            teacher_model = resnet.ResNet110()
            teacher_checkpoint = 'experiments/base_resnet110/best.pth.tar'
            teacher_model = teacher_model.cuda() if params.cuda else teacher_model
            student_checkpoint = student_checkpoint + 'resnet110_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)

        elif params.teacher == "fresnet110":
            #teacher_model = models.resnet18(num_classes=200)#,pretrained=True)
            teacher_checkpoint = 'experiments/base_fresnet110/best.pth.tar'
            #teacher_model.load_state_dict(teacher_checkpoint())
            #utils.load_checkpoint(teacher_checkpoint, teacher_model)
            #Finetune Final few layers to adjust for tiny imagenet input
            #teacher_model.avgpool = nn.AdaptiveAvgPool2d(1)
            #num_ftrs = teacher_model.fc.in_features
            #teacher_model.fc = nn.Linear(num_ftrs, 200, bias=True)
            #teacher_model.fc.weight = nn.Linear(200, 512)
            #teacher_model = teacher_model.to(device)
            #teacher_model = teacher_model.cuda() if params.cuda else teacher_model
            student_checkpoint = student_checkpoint + 'fresnet110_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)
            #teacher_model.load_state_dict(torch.load("/home/kysunami/mygit/timagenet-kd-1/experiments/base_fresnet110/resnet18-5c106cde.pth"))
            #opt = parser.parse_args()
            #if opt.origtrain !="":
                #state_d= torch.load(opt.origtrain)
                #print("\n keys--> ",state_d.keys())
                #teacher_model.load_state_dict(torch.load(opt.origtrain))
            #Load Resnet18
            teacher_model = models.resnet18(num_classes=200)
            #Finetune Final few layers to adjust for tiny imagenet input
            teacher_model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
            teacher_model.maxpool = nn.Sequential()
            teacher_model.avgpool = nn.AdaptiveAvgPool2d(1)
            teacher_model.fc.out_features = 200
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #teacher_model = teacher_model.to(device)
            teacher_model = teacher_model.cuda() if params.cuda else teacher_model
            pretrained_dict = torch.load('experiments/base_fresnet110/resnet18-5c106cde.pth')
            teacher_model_dict = teacher_model.state_dict()
            first_layer_weight = teacher_model_dict['conv1.weight']
            pretrained_dict = {b[0]:b[1] for a,b in zip(teacher_model_dict.items(), pretrained_dict.items()) if a[1].size() == b[1].size()}
            #teacher_model.update(pretrained_dict) 
            teacher_model.load_state_dict(teacher_model_dict)


        elif params.teacher == "preresnet110":
            teacher_model = preresnet.PreResNet(depth=110, num_classes=200)
            teacher_checkpoint = 'experiments/base_preresnet110/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()
            student_checkpoint = student_checkpoint + 'preresnet_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)

        elif params.teacher == "pyramidnet":
            teacher_model = pyramidnet.PyramidNet(dataset='cifar10', depth=272, alpha=200, num_classes=200, bottleneck=True)
            teacher_checkpoint = 'experiments/base_pyramidnet/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()
            student_checkpoint = student_checkpoint + 'pyramidnet_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)

        elif params.teacher == "cnn":
            teacher_model = net.Net(params)
            teacher_checkpoint = 'experiments/base_cnn/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()
            student_checkpoint = student_checkpoint + 'cnn_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)

        elif params.teacher == "lenet5":
            teacher_model = lenet5.LeNet5(channels=3, class_count=200, act='relu')
            teacher_checkpoint = 'experiments/base_lenet5/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()
            student_checkpoint = student_checkpoint + 'lenet5_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)

        elif params.teacher == "lenet5m":
            teacher_model = lenet5m.LeNet5m()
            teacher_checkpoint = 'experiments/base_lenet5m/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()
            student_checkpoint = student_checkpoint + 'lenet5m_teacher/dropout0.0/best.pth.tar'

        elif params.teacher == "lenet5n":
            teacher_model = lenet5n.LeNet5n(channels=3, class_count=200, act='relu')
            teacher_checkpoint = 'experiments/base_lenet5n/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()
            student_checkpoint = student_checkpoint + 'lenet5n_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)

        elif params.teacher == "lenet5n6":
            teacher_model = lenet5n6.LeNet5n6(channels=3, class_count=200, act='relu')
            teacher_checkpoint = 'experiments/base_lenet5n6/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()
            student_checkpoint = student_checkpoint + 'lenet5n6_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)

        elif params.teacher == "lenet5n10":
            teacher_model = lenet5n10.LeNet5n10(channels=3, class_count=200, act='relu')
            teacher_checkpoint = 'experiments/base_lenet5n10/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()
            student_checkpoint = student_checkpoint + 'lenet5n10_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)

        elif params.teacher == "lenet5n1":
            teacher_model = lenet5n1.LeNet5n1(channels=3, class_count=200, act='relu')
            teacher_checkpoint = 'experiments/base_lenet5n1/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()
            student_checkpoint = student_checkpoint + 'lenet5n1_teacher/dropout0.0/best.pth.tar'
            print(student_checkpoint)

        #utils.load_checkpoint(teacher_checkpoint, teacher_model)
        if os.path.isfile(student_checkpoint):
            utils.load_checkpoint(student_checkpoint, model)#karuna to add student checkpoints

        # Train the model with KD
        logging.info("Experiment - model version: {}".format(params.model_version))
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        logging.info("First, loading the teacher model and computing its outputs...")
        train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd,
                              metrics, params, args.model_dir, args.restore_file)

    # non-KD mode: regular training of the baseline CNN or ResNet-18
    else:
        if params.model_version == "cnn":
            model = net.Net(params).cuda() if params.cuda else net.Net(params)
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            #optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            # fetch loss function and metrics
            loss_fn = net.loss_fn
            metrics = net.metrics
            utils.load_checkpoint('experiments/base_cnn/best.pth.tar', model)

        elif params.model_version == "lenet5":
            model = lenet5.LeNet5(channels=3, class_count=200, act='relu').cuda() if params.cuda else lenet5.LeNet5(channels=3, class_count=200, act='relu')
            #optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            #optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            # fetch loss function and metrics
            loss_fn = lenet5.loss_fn
            metrics = lenet5.metrics
            utils.load_checkpoint('experiments/base_lenet5/best.pth.tar', model)

        elif params.model_version == "lenet5m":
            model = lenet5m.LeNet5m().cuda() if params.cuda else lenet5m.LeNet5m()
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics
            loss_fn = lenet5m.loss_fn
            metrics = lenet5m.metrics
            utils.load_checkpoint('experiments/base_lenet5m/best.pth.tar', model)

        elif params.model_version == "lenet5n":
            model = lenet5n.LeNet5n(channels=3, class_count=200, act='relu').cuda() if params.cuda else lenet5n.LeNet5n(channels=3, class_count=200, act='relu')
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics
            loss_fn = lenet5n.loss_fn
            metrics = lenet5n.metrics
            utils.load_checkpoint('experiments/base_lenet5n/best.pth.tar', model)

        elif params.model_version == "lenet5n6":
            model = lenet5n6.LeNet5n6(channels=3, class_count=200, act='relu').cuda() if params.cuda else lenet5n6.LeNet5n6(channels=3, class_count=200, act='relu')
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics
            loss_fn = lenet5n6.loss_fn
            metrics = lenet5n6.metrics
            utils.load_checkpoint('experiments/base_lenet5n6/best.pth.tar', model)

        elif params.model_version == "lenet5n10":
            model = lenet5n10.LeNet5n10(channels=3, class_count=200, act='relu').cuda() if params.cuda else lenet5n10.LeNet5n10(channels=3, class_count=200, act='relu')
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics
            loss_fn = lenet5n10.loss_fn
            metrics = lenet5n10.metrics
            utils.load_checkpoint('experiments/base_lenet5n10/best.pth.tar', model)

        elif params.model_version == "lenet5n1":
            model = lenet5n1.LeNet5n1(channels=3, class_count=200, act='relu').cuda() if params.cuda else lenet5n1.LeNet5n1(channels=3, class_count=200, act='relu')
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics
            loss_fn = lenet5n1.loss_fn
            metrics = lenet5n1.metrics
            utils.load_checkpoint('experiments/base_lenet5n1/best.pth.tar', model)

        elif params.model_version == "resnet18":
            model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            #optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            # fetch loss function and metrics
            loss_fn = resnet.loss_fn
            metrics = resnet.metrics
            utils.load_checkpoint('experiments/base_resnet18/best.pth.tar', model)

        elif params.model_version == "resnet50":
            model = resnet.ResNet50().cuda() if params.cuda else resnet.ResNet50()
            #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            #optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics
            loss_fn = resnet.loss_fn
            metrics = resnet.metrics
            #utils.load_checkpoint('experiments/base_resnet50/best.pth.tar', model)

        elif params.model_version == "preresnet110":
            model = net.Net(params).cuda() if params.cuda else net.Net(params)
            #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            #optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics
            loss_fn = preresnet.loss_fn
            metrics = preresnet.metrics
            utils.load_checkpoint('experiments/base_preresnet/best.pth.tar', model)

        elif params.model_version == "pyramidnet":
            model = pyramidnet.PyramidNet(dataset='cifar10', depth=272, alpha=200, num_classes=200, bottleneck=True).cuda() if params.cuda else pyramidnet.PyramidNet(dataset='cifar10', depth=272, alpha=200, num_classes=200, bottleneck=True)
            #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            #optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics
            loss_fn = pyramidnet.loss_fn
            metrics = pyramidnet.metrics
            utils.load_checkpoint('experiments/base_pyramidnet/best.pth.tar', model)

        elif params.model_version == "fresnet110":
            model = models.resnet18(num_classes=200).cuda() if params.cuda else models.resnet18(num_classes=200)
            #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            #optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics
            loss_fn = resnet.loss_fn
            metrics = resnet.metrics
            utils.load_checkpoint('experiments/base_fresnet110/best.pth.tar', model)

        elif params.model_version == "resnet110":
            model = resnet.ResNet110().cuda() if params.cuda else resnet.ResNet110()
            #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            #optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
            #optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics
            loss_fn = resnet.loss_fn
            metrics = resnet.metrics
            #try:
                #if os.system(utils.load_checkpoint('experiments/base_resnet110/best.pth.tar', model)) != 0:
                    #pass
            #catch:
                #pass

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, params,
                           args.model_dir, args.restore_file)
