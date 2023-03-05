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
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
#from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
tb_logger = loggers.TensorBoardLogger('logs/')

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
        

    def training_step(self, batch, batch_idx):
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
        
        
def main():
    # init model
    autoencoder = LitAutoEncoder()
    trainer = pl.Trainer(automatic_optimization=False,
                                         max_epochs=30,
                                         weights_summary='full',
                                         logger=tb_logger,
                                         gpus=1,
                                         precision=16)

    #trainer.fit(autoencoder, train_loader)
    trainer.test()
    
    PATH = '/home/arun/workspace/pytorch_posenet/light_posenet/logs/default/version_0/checkpoints/epoch=25-step=198223.ckpt'
    model = LitAutoEncoder.load_from_checkpoint(PATH)
    input_sample = torch.Tensor(1, 3, 256, 256)
    out = model(input_sample)
    print(out)
    
    filepath = 'model_lightning.onnx'
    model.to_onnx(filepath, input_sample, export_params=True)

    '''ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: np.random.randn(1, 64).astype(np.float16)}
    ort_outs = ort_session.run(None, ort_inputs)'''

 
if __name__ == '__main__':
     main()
    
          
