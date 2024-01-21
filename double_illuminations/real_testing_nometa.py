import math
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from models.network_meta import aggregate_net
from datasets.Hyperspectral_dataset import Training_dataset, Testing_dataset, Real_dataset
from torch.utils.tensorboard import SummaryWriter

from engine import train_one_epoch, evaluate, real_evaluate
from models.loss import reconstruct_loss
import utils.misc as utils


def get_args_parser():
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--meta_lr', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--num_images', default=2, type=int,
                        help='number of input RGB images')

    parser.add_argument('--num_gradient', default=5, type=int)

    # dataset parameters
    parser.add_argument('--dataset_path', default='/data/dhuo/exp_data/spectral_dataset')

    parser.add_argument('--output_dir', default='checkpoints/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--result_dir', default='real_results/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--log_dir', default='logs/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--resume', default='checkpoints/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--eval', default=True, action='store_true')

    # parser.add_argument('--resume', help='resume from checkpoint')
    # parser.add_argument('--eval', action='store_true')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--num_workers', default=4, type=int)

    return parser


def main(args):
    device = torch.device(args.device)

    torch.backends.cuda.matmul.allow_tf32 = False

    writer = SummaryWriter(args.log_dir)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_train = Training_dataset(img_folder=os.path.join(args.dataset_path, "training_patches128"))
    dataset_val = Real_dataset(img_folder=os.path.join(args.dataset_path, "real_testing"))

    train_data_loader = DataLoader(dataset=dataset_train,
                                   num_workers=args.num_workers,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=False)

    val_loader = DataLoader(dataset=dataset_val,
                            num_workers=1,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=False)

    criterion = reconstruct_loss()
    output_dir = Path(args.output_dir)

    model = aggregate_net(args.num_images * 3, 31).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param = list(model.parameters())

    print('number of params:', n_parameters)

    optimizer = torch.optim.Adam(param,
                                 # [{'params': param}, {'params': model.scale_factors, 'lr': args.lr}],
                                 lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    per_epoch_iteration = len(train_data_loader) // 4
    total_iteration = per_epoch_iteration * args.epochs

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

    if args.resume:

        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        mean_MRAE, mean_RMSE = real_evaluate(model, val_loader, device, args.result_dir)
        print("mean_MRAE: " + str(mean_MRAE))
        print("mean_RMSE: " + str(mean_RMSE))
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    #
    # if args.log_dir:
    #     Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    main(args)
