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
scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter()


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader

'''
torch.manual_seed(0)
np.random.seed(0)
torch.set_deterministic(True)'''

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader

def tf_learn(model):
    modules = list(model.children())
    #modules[0]: Base, modules[1]: Cpm, modules[2]:InitialStage, modules[3]:RefinementStage
    for block in range(len(modules)):
        if block in range(0, 2):
            for param in modules[block].parameters():
                param.requires_grad = False
                   
        if block==3:
            layers = []
            for block_layers in modules[block][0].children():
                for layer in  block_layers.children():
                    layers.append(layer)
                    
            for layer in layers[0:2]:
                for param in layer.parameters():
                    param.requires_grad = False                    
    return model
      
      
def train(prepared_train_labels, train_images_folder, 
          num_refinement_stages, base_lr, batch_size, batches_per_iter,
          num_workers, checkpoint_path, weights_only, from_mobilenet, checkpoints_folder, log_after,
          val_labels, val_images_folder, val_output_name, checkpoint_after, val_after):
    net = PoseEstimationWithMobileNet(num_refinement_stages)
    net = tf_learn(net)
           
    stride = 8
    sigma = 7
    path_thickness = 1
    dataset = CocoTrainDataset(prepared_train_labels, train_images_folder,
                               stride, sigma, path_thickness,
                               transform=transforms.Compose([
                                   ConvertKeypoints(),
                                   Scale(),
                                   Rotate(pad=(128, 128, 128)),
                                   CropPad(pad=(128, 128, 128)),
                                   Flip()]))
                                       
    batch_size = 20
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    validation_split = 0.25
    num_iter = 0
    current_epoch = 0
    epochs = 60
    drop_after_epoch = [100, 200, 260]
    
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                            sampler=train_sampler, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                            sampler=valid_sampler, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)

    optimizer = optim.Adam([
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

    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)

        if from_mobilenet:
            load_from_mobilenet(net, checkpoint)
        else:
            load_state(net, checkpoint)
            if not weights_only:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                num_iter = checkpoint['iter']
                current_epoch = checkpoint['current_epoch']

    scaler = torch.cuda.amp.GradScaler()
    net = DataParallel(net).cuda()

    train_losses = []
    val_losses_ = []
    
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train()  # Set model to training mode
        else:
            optimizer.zero_grad() #zero the parameter gradients
            net.eval()   # Set model to evaluate mode
            
        for epochId in range(current_epoch, epochs):
            running_loss_train = 0.0
            running_loss_val = 0.0
            
            scheduler.step()
            total_losses_train = [0, 0] * (num_refinement_stages + 1)  # heatmaps loss, paf loss per stage
            total_losses_val = [0, 0] * (num_refinement_stages + 1)
            batch_per_iter_idx = 0
            
            if phase == 'train':
                #batch_data = next(iter(train_loader))
                net.train()
                for batch_data in train_loader:
                    if batch_per_iter_idx == 0:
                        optimizer.zero_grad()

                    for i in batch_data:
                        images = i['image']
                        keypoint_masks = i['keypoint_mask']
                        paf_masks = i['paf_mask']
                        keypoint_maps = i['keypoint_maps']
                        paf_maps = i['paf_maps']
                        
                        images = np.expand_dims(images, axis=0)
                        keypoint_masks = np.expand_dims(keypoint_masks, axis=0)
                        paf_masks = np.expand_dims(paf_masks, axis=0)
                        keypoint_maps = np.expand_dims(keypoint_maps, axis=0)
                        paf_maps = np.expand_dims(paf_maps, axis=0)
                        
                        images = torch.from_numpy(images).cuda()
                        keypoint_masks = torch.from_numpy(keypoint_masks).cuda()
                        paf_masks = torch.from_numpy(paf_masks).cuda()
                        keypoint_maps = torch.from_numpy(keypoint_maps).cuda()
                        paf_maps = torch.from_numpy(paf_maps).cuda()
                        
                        stages_output = net(images)

                        losses = []
                        for loss_idx in range(len(total_losses_train) // 2):
                            losses.append(l2_loss(stages_output[loss_idx * 2], 
                                           keypoint_maps, keypoint_masks, images.shape[0]))
                            losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, 
                                           paf_masks, images.shape[0]))
                            total_losses_train[loss_idx * 2] += losses[-2].item() / batches_per_iter
                            total_losses_train[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter

                        with torch.cuda.amp.autocast():
                            loss = losses[0]
                            for loss_idx in range(1, len(losses)):
                                loss += losses[loss_idx]
                            loss /= batches_per_iter
                            #loss.backward()
                            scaler.scale(loss).backward()
                            running_loss_train += loss.item() * images.size(0)
                            
                            batch_per_iter_idx += 1
                            if batch_per_iter_idx == batches_per_iter:
                                #optimizer.step()
                                scaler.step(optimizer)
                                batch_per_iter_idx = 0
                                num_iter += 1
                                scaler.update()
                                
                                writer.add_scalar('Loss/train vs epochId  ', loss, epochId)
                                writer.add_scalar('Loss/train vs Num_iter', loss, num_iter)
                            else:
                                continue

                            if num_iter % log_after == 0:
                                print('Iter: {}, epochId: {}'.format(num_iter, epochId))
                                for loss_idx in range(len(total_losses_train) // 2):
                                    print('\n'.join(['stage{}_pafs_loss: {}', 
                                        'stage{}_heatmaps_loss: {}']).format(
                                        loss_idx + 1, total_losses_train[loss_idx * 2 + 1] / log_after,
                                        loss_idx + 1, total_losses_train[loss_idx * 2] / log_after))
                                for loss_idx in range(len(total_losses_train)):
                                    total_losses_train[loss_idx] = 0
                            if num_iter % checkpoint_after == 0:
                                snapshot_name = '{}/checkpoint_iter_{}_epochId_{}.pth'.format(
                                                           checkpoints_folder,
                                                           num_iter, epochId )
                                torch.save({'state_dict': net.module.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            'scheduler': scheduler.state_dict(),
                                            'iter': num_iter,
                                            'current_epoch': epochId},
                                           snapshot_name)
                            
            epoch_loss = running_loss_train / len(train_loader)
            train_losses.append(epoch_loss)
            
            elif phase == 'val':
                net.eval()
                #batch_data_val = next(iter(validation_loader))
                for batch_data_val in validation_loader:
                    if batch_per_iter_idx == 0:
                        optimizer.zero_grad()
                    
                    for i in batch_data_val:
                        images = i['image']
                        keypoint_masks = i['keypoint_mask']
                        paf_masks = i['paf_mask']
                        keypoint_maps = i['keypoint_maps']
                        paf_maps = i['paf_maps']
                        
                        images = np.expand_dims(images, axis=0)
                        keypoint_masks = np.expand_dims(keypoint_masks, axis=0)
                        paf_masks = np.expand_dims(paf_masks, axis=0)
                        keypoint_maps = np.expand_dims(keypoint_maps, axis=0)
                        paf_maps = np.expand_dims(paf_maps, axis=0)
                        
                        images = torch.from_numpy(images).cuda()
                        keypoint_masks = torch.from_numpy(keypoint_masks).cuda()
                        paf_masks = torch.from_numpy(paf_masks).cuda()
                        keypoint_maps = torch.from_numpy(keypoint_maps).cuda()
                        paf_maps = torch.from_numpy(paf_maps).cuda()

                        stages_output = net(images)

                        val_losses = []
                        for loss_idx in range(len(total_losses_val) // 2):
                            val_losses.append(l2_loss(stages_output[loss_idx * 2], 
                                       keypoint_maps, keypoint_masks, images.shape[0]))
                            val_losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, 
                                       paf_masks, images.shape[0]))
                            total_losses_val[loss_idx * 2] += val_losses[-2].item() / batches_per_iter
                            total_losses_val[loss_idx * 2 + 1] += val_losses[-1].item() / batches_per_iter

                        with torch.cuda.amp.autocast():
                            val_loss = val_losses[0]
                            for loss_idx in range(1, len(val_losses)):
                                val_loss += val_losses[loss_idx]
                            val_loss /= batches_per_iter
                            #val_loss.backward()
                            #scaler.scale(val_loss).backward()
                            scaler.scale(val_loss)
                        
                            running_loss_val += val_loss.item() * images.size(0)
                            optimizer.zero_grad() 
                            
                            batch_per_iter_idx += 1
                            if batch_per_iter_idx == batches_per_iter:
                                #optimizer.step()
                                #scaler.step(optimizer)
                                batch_per_iter_idx = 0
                                num_iter += 1
                                scaler.update()
                                
                                writer.add_scalar('Loss/Val vs EpochId', val_loss, epochId)
                                writer.add_scalar(' Loss/Val vs Num_iter', val_loss, num_iter)
                            else:
                                continue

                            if num_iter % log_after == 0:
                                print('Iter: {}, epochId: {}'.format(num_iter, epochId))
                                for loss_idx in range(len(total_losses_val) // 2):
                                    print('\n'.join(['stage{}_pafs_loss: {}', 
                                        'stage{}_heatmaps_loss: {}']).format(
                                        loss_idx + 1, total_losses_val[loss_idx * 2 + 1] / log_after,
                                        loss_idx + 1, total_losses_val[loss_idx * 2] / log_after))
                                for loss_idx in range(len(total_losses_val)):
                                    total_losses_val[loss_idx] = 0
                        
            epoch_loss = running_loss_val / len(validation_loader)
            val_losses_.append(epoch_loss)    
                    
    plt.plot(range(current_epoch, epochs), train_losses, 'b', label='Training Loss')
    plt.plot(range(current_epoch, epochs), val_losses_, 'r', label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y%H_%M_%S")
    
    plt.savefig("loss_plots/latest/Loss.svg")
    plt.savefig("loss_plots/all/Loss"+str(dt_string)+'.svg')
    plt.show()
        
    writer.flush()
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-train-labels', type=str, required=True,
                        help='path to the file with prepared annotations')
    parser.add_argument('--train-images-folder', type=str, required=True, help='path to COCO train images folder')
    parser.add_argument('--num-refinement-stages', type=int, default=1, help='number of refinement stages')
    parser.add_argument('--base-lr', type=float, default=4e-5, help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=5, help='batch size')
    parser.add_argument('--batches-per-iter', type=int, default=1, help='number of batches to accumulate gradient from')
    parser.add_argument('--num-workers', type=int, default=1, help='number of workers')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint to continue training from')
    parser.add_argument('--from-mobilenet', action='store_true',
                        help='load weights from mobilenet feature extractor')
    parser.add_argument('--weights-only', action='store_true',
                        help='just initialize layers with pre-trained weights and start training from the beginning')
    parser.add_argument('--experiment-name', type=str, default='default',
                        help='experiment name to create folder for checkpoints')
    parser.add_argument('--log-after', type=int, default=100, help='number of iterations to print train loss')

    parser.add_argument('--val-labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--val-images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--val-output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--checkpoint-after', type=int, default=5000,
                        help='number of iterations to save checkpoint')
    parser.add_argument('--val-after', type=int, default=5000,
                        help='number of iterations to run validation')
    args = parser.parse_args()

    checkpoints_folder = '{}_checkpoints'.format(args.experiment_name)
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    train(args.prepared_train_labels, args.train_images_folder, args.num_refinement_stages, args.base_lr, args.batch_size,
          args.batches_per_iter, args.num_workers, args.checkpoint_path, args.weights_only, args.from_mobilenet,
          checkpoints_folder, args.log_after, args.val_labels, args.val_images_folder, args.val_output_name,
          args.checkpoint_after, args.val_after)
