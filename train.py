import argparse
import logging
import sys
import os
from pathlib import Path
from os import listdir
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset, Dataset_Pro
from utils.dice_score import dice_loss
from evaluate import evaluate
from Models.Treble_Former import Treble_Former



def MHC_Ensemble(midice):

    nor_dice = [midice[0], midice[1], midice[2], midice[3], midice[4]]

    mid_nor_B = torch.tensor([nor_dice[0], nor_dice[1], nor_dice[2]]).float()
    mid_nor_C = torch.tensor([nor_dice[0], nor_dice[1], nor_dice[2], nor_dice[3], nor_dice[4]]).float()
    B_acc = 1
    C_acc = 1

    Combine_B = torch.tensor([mid_nor_B[0] * B_acc, mid_nor_B[1] * B_acc, mid_nor_B[2] * B_acc]).float()
    Combine_C = torch.tensor([mid_nor_C[0] * C_acc, mid_nor_C[1] * C_acc, mid_nor_C[2] * C_acc,
                            mid_nor_C[3] * C_acc, mid_nor_C[4] * C_acc]).float()

    Weight_A = torch.tensor([1.0])
    Weight_B = F.softmax(Combine_B, dim=0)
    Weight_C = F.softmax(Combine_C, dim=0)

    Weight = torch.tensor([Weight_C[0] + Weight_B[0] + Weight_A[0], Weight_C[1] + Weight_B[1], Weight_C[2] + Weight_B[2], Weight_C[3], Weight_C[4]])

    return [Weight[0], Weight[1], Weight[2], Weight[3], Weight[4]]


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              test_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    dir_img = Path('./data/imgs/')
    dir_mask = Path('./data/masks/')
    dir_checkpoint = Path('./checkpoints/')
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    list_img = listdir(dir_img)
    list_mask = listdir(dir_mask)
    n_val = int(len(list_img) * val_percent)
    n_test = int(len(list_img) * test_percent)
    n_train = int(len(list_img) - n_val - n_test)
    train_img_set, val_img_set, test_img_set = random_split(list_img, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))
    train_mask_set, val_mask_set, test_mask_set = random_split(list_mask, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))
    train_set = Dataset_Pro(dir_img, dir_mask, train_img_set, train_mask_set, img_scale, augmentations=True)
    val_set = Dataset_Pro(dir_img, dir_mask, val_img_set, val_mask_set, img_scale, augmentations=False)
    test_set = Dataset_Pro(dir_img, dir_mask, test_img_set, test_mask_set, img_scale, augmentations=False)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Testing size:    {n_test}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-8, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, min_lr=1e-6, patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    weight = [1.5, 0.5, 0.5, 0.25, 0.25]
    midice = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    weight = MHC_Ensemble(midice)
                    deep_out = net(images)
                    masks_pred = deep_out[0] * weight[0] + deep_out[1] * weight[1] + deep_out[2] * weight[2] + \
                                 deep_out[3] * weight[3] + deep_out[4] * weight[4]
                    true_masks = F.one_hot(true_masks.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()
                    loss = criterion(deep_out[0], true_masks) + \
                           dice_loss(F.softmax(deep_out[0], dim=1).float(), true_masks, multiclass=True) + \
                           criterion(deep_out[1], true_masks) + \
                           dice_loss(F.softmax(deep_out[1], dim=1).float(), true_masks, multiclass=True) + \
                           criterion(deep_out[2], true_masks) + \
                           dice_loss(F.softmax(deep_out[2], dim=1).float(), true_masks,multiclass=True) + \
                           criterion(deep_out[3], true_masks) + \
                           dice_loss(F.softmax(deep_out[3], dim=1).float(), true_masks, multiclass=True) + \
                           criterion(deep_out[4], true_masks) + \
                           dice_loss(F.softmax(deep_out[4], dim=1).float(), true_masks, multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (1 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:

                        val_score, midice = evaluate(net, weight, val_loader, str('val'), device)
                        dice, mdice = evaluate(net, weight, test_loader, str('test'), device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        logging.info('Testing Dice score: {}'.format(dice))
                        logging.info('Testing mDice score: {}'.format(mdice))

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=15, help='Number of epochs')#default=150
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=[352, 352], help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--testing', '-t', dest='test', type=float, default=10.0,
                        help='Percent of the data that is used as testing (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = Treble_Former(n_channels=3, n_classes=2)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  test_percent=args.test / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
