import math
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
from datasets.Hyperspectral_dataset import Training_dataset, Testing_dataset
from torch.utils.tensorboard import SummaryWriter

from meta_engine import meta_train_one_epoch, meta_evaluate
from models.loss import meta_reconstruct_loss, reconstruct_loss
import utils.misc as utils


def get_args_parser():
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--meta_lr', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--num_images', default=1, type=int,
                        help='number of input RGB images')

    parser.add_argument('--num_gradient', default=5, type=int)

    # dataset parameter
    parser.add_argument('--dataset_path', default='/data/dhuo/exp_data/spectral_dataset')

    parser.add_argument('--output_dir', default='meta_checkpoints/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--result_dir', default='meta_results/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--log_dir', default='logs/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    # parser.add_argument('--resume', default='meta_checkpoints/checkpoint.pth', help='resume from checkpoint')
    # parser.add_argument('--eval', default=True, action='store_true')

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')

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
    dataset_val = Testing_dataset(img_folder=os.path.join(args.dataset_path, "testing"))

    train_data_loader = DataLoader(dataset=dataset_train,
                                   num_workers=args.num_workers,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=False,
                                   drop_last=True)

    val_loader = DataLoader(dataset=dataset_val,
                            num_workers=1,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=False)

    train_criterion = reconstruct_loss()

    test_criterion = meta_reconstruct_loss()
    output_dir = Path(args.output_dir)

    model = aggregate_net(args.num_images * 3, 31).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param = list(model.parameters())

    print('number of params:', n_parameters)

    primary_optimizer = torch.optim.SGD(
        list(model.sensitivity_net.parameters()) + list(model.pyramid_net.parameters()), lr=args.meta_lr)

    auxiliary_optimizer = torch.optim.SGD(list(model.auxiliary_net.parameters()), lr=args.meta_lr)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #     primary_optimizer.load_state_dict(checkpoint['primary_optimizer'])
        #     auxiliary_optimizer.load_state_dict(checkpoint['auxiliary_optimizer'])
        # args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        '''
        meta_evaluate(model: torch.nn.Module, criterion: torch.nn.Module,
                  data_loader: Iterable, primary_optimizer: torch.optim.Optimizer,
                  auxiliary_optimizer: torch.optim.Optimizer,
                  device: torch.device, num_gradient, result_dir, max_norm: float = 0):
        
        '''

        mean_MRAE, mean_RMSE, mean_SAM = meta_evaluate(model, test_criterion, val_loader, primary_optimizer,
                                                       auxiliary_optimizer,
                                                       device, args.num_gradient, args.result_dir)
        print("mean_MRAE: " + str(mean_MRAE))
        print("mean_RMSE: " + str(mean_RMSE))
        print("mean_SAM: " + str(mean_SAM))
        return

    print("Start training")
    start_time = time.time()

    min_mean_MRAE = math.inf
    min_mean_RMSE = math.inf
    min_mean_SAM = math.inf

    for epoch in range(args.start_epoch, args.epochs):

        meta_train_one_epoch(model, train_criterion, train_data_loader, primary_optimizer, auxiliary_optimizer, device,
                             epoch, writer, args.batch_size, args.num_gradient)

        if (epoch + 1) % 1 == 0:
            mean_MRAE, mean_RMSE, mean_SAM = meta_evaluate(model, test_criterion, val_loader, primary_optimizer,
                                                           auxiliary_optimizer, device, args.num_gradient,
                                                           args.result_dir)

            writer.add_scalar('testing/mean_MRAE', mean_MRAE, epoch)
            writer.add_scalar('testing/mean_RMSE', mean_RMSE, epoch)
            writer.add_scalar('testing/mean_SAM', mean_SAM, epoch)

            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if min_mean_MRAE > mean_MRAE or min_mean_RMSE > mean_RMSE or min_mean_SAM > mean_SAM:
                    min_mean_MRAE = mean_MRAE
                    min_mean_RMSE = mean_RMSE
                    min_mean_SAM = mean_SAM
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'network': [model],
                        'primary_optimizer': primary_optimizer.state_dict(),
                        'auxiliary_optimizer': auxiliary_optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    main(args)
