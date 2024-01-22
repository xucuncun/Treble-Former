import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, weight, dataloader, mod, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    mdice_score = 0
    mdice1, mdice2, mdice3, mdice4, mdice5 = 0, 0, 0, 0, 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            deep_out = net(image)
            mask_pred = deep_out[0] * weight[0] + deep_out[1] * weight[1] + deep_out[2] * weight[2] + \
                        deep_out[3] * weight[3] + deep_out[4] * weight[4]
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                mdice_score += multiclass_dice_coeff(mask_pred[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)

            if mod == str('val'):
                deep_out1 = F.one_hot(deep_out[0].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice1 += multiclass_dice_coeff(deep_out1[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)
                deep_out2 = F.one_hot(deep_out[1].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice2 += multiclass_dice_coeff(deep_out2[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)
                deep_out3 = F.one_hot(deep_out[2].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice3 += multiclass_dice_coeff(deep_out3[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)
                deep_out4 = F.one_hot(deep_out[3].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice4 += multiclass_dice_coeff(deep_out4[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)
                deep_out5 = F.one_hot(deep_out[4].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice5 += multiclass_dice_coeff(deep_out5[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score

    if num_val_batches == 0:
        dice = dice_score
        mdice = mdice_score
        midice = [mdice1, mdice2, mdice3, mdice4, mdice5]
    else:
        dice = dice_score / num_val_batches
        mdice = mdice_score / num_val_batches
        midice = [mdice1 / num_val_batches, mdice2 / num_val_batches, mdice3 / num_val_batches,
                  mdice4 / num_val_batches, mdice5 / num_val_batches]

    if mod == str('val'):
        return dice, midice
    else:
        return dice, mdice
