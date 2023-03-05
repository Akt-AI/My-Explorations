import argparse
import cv2
import os
from datetime import datetime

import time
import numpy as np
import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.coco import CocoTrainDataset
from datasets.transformations import ConvertKeypoints, Scale, Rotate, CropPad, Flip
from modules.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.loss import l2_loss
from modules.load_state import load_state, load_from_mobilenet
from val import evaluate

from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
#from config import *

#/home/arun/workspace/pytorch_posenet/env3.6/lib/python3.6/site-packages/pytorch_lightning/trainer/connectors/logger_connector.py, Line No 549 commented

'''
5.6 Lifecycle
The methods in the LightningModule are called in this order:
1. __init__()
2. prepare_data()
3. configure_optimizers()
4. train_dataloader()
If you define a validation loop then
5. val_dataloader()
And if you define a test loop:
6. test_dataloader()
Note: test_dataloader() is only called with .test()
'''

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
#early_stopping = EarlyStopping('val_loss')
#trainer = Trainer(early_stop_callback=early_stopping)

from pytorch_lightning.callbacks import ModelCheckpoint
# saves checkpoints to 'my/path/' whenever 'val_loss' has a new min
#checkpoint_callback = ModelCheckpoint(filepath='my/path/')
#trainer = Trainer(checkpoint_callback=checkpoint_callback)
# save epoch and val_loss in name
# saves a file like: my/path/sample-mnist_epoch=02_val_loss=0.32.ckpt
#checkpoint_callback = ModelCheckpoint(filepath='my/path/sample-mnist_{epoch:02d}-{val_loss:.2f}')

from pytorch_lightning import Trainer
from pytorch_lightning import loggers
#from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
tb_logger = loggers.TensorBoardLogger('logs/')
#trainer = Trainer(logger=tb_logger)

#tb_logger = loggers.TensorBoardLogger('logs/')
#comet_logger = loggers.CometLogger(save_dir='logs/')
#trainer = Trainer(logger=[tb_logger, comet_logger])


writer = SummaryWriter()
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader
random_seed= 42


#Step 1: Define LightningModule
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        num_refinement_stages = 1
        #self.num_refinement_stages = num_refinement_stages
        self.model = PoseEstimationWithMobileNet(num_refinement_stages)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        model = self.model(x)
        return model
        
    '''def train_dataloader(self):
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
            mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)

            return DataLoader(mnist_train, batch_size=64)

            #mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
            #mnist_test = MNIST(os.getcwd(), train=False, download=True,transform=transform)
            # train/val split
            mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
            # assign to use in dataloaders
            self.train_dataset = mnist_train
            self.val_dataset = mnist_val
            self.test_dataset = mnist_test
        
        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=64)
        def val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size=64)
        def test_dataloader(self):
            return DataLoader(self.test_dataset, batch_size=64)
            
        
        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            return {'val_loss': loss}
            
        def validation_epoch_end(self, outputs):
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            return {'val_loss': avg_loss, 'log': tensorboard_logs}
            
        def val_dataloader(self):
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
            mnist_train = MNIST(os.getcwd(), train=True, download=False,
            transform=transform)
            _, mnist_val = random_split(mnist_train, [55000, 5000])
            mnist_val = DataLoader(mnist_val, batch_size=64)
            
            return mnist_val
        
        def test_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            return {'val_loss': loss}
        
        def test_epoch_end(self, outputs):
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            return {'val_loss': avg_loss, 'log': tensorboard_logs}
        def test_dataloader(self):
            transform=transforms.Compose([transforms.ToTensor(), 
                         transforms.Normalize((0.1307,), (0.3081,))])
            mnist_train = MNIST(os.getcwd(), train=False, download=False, transform=transform)
            _, mnist_val = random_split(mnist_train, [55000, 5000])
            mnist_val = DataLoader(mnist_val, batch_size=64)
            return mnist_val
            
        def test_dataloader(self):
            transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,))])
            dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform,
            download=True)
            loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False)
            return loader'''
            

    def training_step(self, batch, batch_idx):
        #current_epoch = 0
        #epochs = 5
        
        num_iter = 0
        drop_after_epoch = [100, 200, 260]
        train_losses = []
        val_losses_ = []
        num_refinement_stages = 1
        batches_per_iter = 1
        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                           milestones=drop_after_epoch, gamma=0.333)
        
        scheduler.step()
        total_losses = [0, 0] * (num_refinement_stages + 1)  # heatmaps loss, paf loss per stage
                
        batch_per_iter_idx = batch_idx   
        if batch_per_iter_idx == 0:
            self.optimizer.zero_grad()

        images = batch['image']
        keypoint_masks = batch['keypoint_mask']
        paf_masks = batch['paf_mask']
        keypoint_maps = batch['keypoint_maps']
        paf_maps = batch['paf_maps']
           
        stages_output = self(images)
        
        losses = []
        for loss_idx in range(len(total_losses) // 2):
            losses.append(l2_loss(stages_output[loss_idx * 2], 
                          keypoint_maps, keypoint_masks, images.shape[0]))
            losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))
            total_losses[loss_idx * 2] += losses[-2].item() / batches_per_iter
            total_losses[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter

        loss = losses[0]
        for loss_idx in range(1, len(losses)):
            loss += losses[loss_idx]
        loss /= batches_per_iter
        loss.backward(retain_graph=True)
        loss.backward()
        
        logs = {'loss': loss}
        #self.logger.summary.scalar('loss', loss)
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        net = self.model
        base_lr = 4e-5
        self.optimizer = optim.Adam([
        {'params': get_parameters_conv(net.model, 'weight')},
        {'params': get_parameters_conv_depthwise(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.cpm, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.cpm, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv(net.initial_stage, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.refinement_stages, 'weight'), 'lr': base_lr * 4},
        {'params': get_parameters_conv(net.refinement_stages, 'bias'), 'lr': base_lr * 8, 
                    'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'bias'), 
                          'lr': base_lr * 2, 'weight_decay': 0},], 
                           lr=base_lr, weight_decay=5e-4)

        return self.optimizer
        
        
def main(prepared_train_labels, train_images_folder,
         checkpoint_path, weights_only, checkpoints_folder):
          
    stride = 8
    sigma = 7
    path_thickness = 1
    batch_size = 5
    validation_split = .2
    shuffle_dataset = True
    num_workers = 8

    dataset = CocoTrainDataset(prepared_train_labels, train_images_folder,
                                   stride, sigma, path_thickness,
                                   transform=transforms.Compose([
                                       ConvertKeypoints(),
                                       Scale(),
                                       Rotate(pad=(128, 128, 128)),
                                       CropPad(pad=(128, 128, 128)),
                                       Flip()]))
                                       
                                                                        
    dataset_size = len(dataset)
    '''indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    #train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                            sampler=valid_sampler, shuffle=False, 
                            num_workers=num_workers, collate_fn=lambda x: x)
                            
                            
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                            sampler=valid_sampler, shuffle=False, 
                            num_workers=num_workers, collate_fn=lambda x: x)'''


    train_loader = DataLoader(dataset)
    # init model
    autoencoder = LitAutoEncoder()
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    #trainer = pl.Trainer(gpus=1)

    trainer = pl.Trainer(automatic_optimization=False,
                                         max_epochs=1,
                                         weights_summary='full',
                                         #overfit_pct=0.01,
                                         #logger=[tb_logger, comet_logger],
                                         logger=tb_logger,
                                         )
                                         #precision=16)
                                         #early_stop_checkpoint=True,
                                         #train_percent_check=0.5,
                                         #val_check_interval=0.25)
    trainer.fit(autoencoder, train_loader)
    #trainer.test()
    '''
    model = LitMNIST.load_from_checkpoint(PATH)
    x = torch.Tensor(1, 1, 28, 28)
    out = model(x)
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-train-labels', type=str, required=True,
                        help='path to the file with prepared annotations')
    parser.add_argument('--train-images-folder', type=str, required=True, help='path to COCO train images folder')
    
    parser.add_argument('--batches-per-iter', type=int, default=1, help='number of batches to accumulate gradient from')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint to continue training from')
    parser.add_argument('--weights-only', action='store_true',
                        help='just initialize layers with pre-trained weights and start training from the beginning')
    parser.add_argument('--experiment-name', type=str, default='default',
                        help='experiment name to create folder for checkpoints')
   
    args = parser.parse_args()

    checkpoints_folder = '{}_checkpoints'.format(args.experiment_name)
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)
        
    main(args.prepared_train_labels, args.train_images_folder, args.checkpoint_path, args.weights_only, checkpoints_folder)

    
          
