from glob import glob
import warnings

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from sklearn.model_selection import KFold, StratifiedKFold

from utils import AverageMeter, get_score
from dataset import FaceDataset
from transform import train_transforms, valid_transforms
from model import VGG_MLP
warnings.simplefilter('ignore')


def train_fn(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).float()

        y_preds = model(imgs).squeeze(1)
        loss = loss_fn(y_preds, image_labels)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), 32)

        if ((step + 1) % 1 == 0) or ((step + 1) == len(train_loader)):
            description = f'epoch {epoch} loss: {losses.avg:.4f}'
            pbar.set_description(description)
        scheduler.step()

    return losses.avg


def valid_fn(epoch, model, loss_fn, val_loader, device, scheduler=None):
    model.eval()
    losses = AverageMeter()
    image_preds_all = []
    image_targets_all = []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).float()

        image_preds = model(imgs).squeeze(1)  # output = model(input)
        image_preds_all += [np.where(image_preds.detach().cpu().numpy() < 0.5, 0, 1)]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)
        losses.update(loss.item(), 32)

        if ((step + 1) % 1 == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {losses.avg:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation class accuracy = {:.4f}'.format(
        (image_preds_all == image_targets_all).mean()))

    return losses.avg, image_preds_all, image_targets_all


if __name__ == '__main__':
    df = pd.read_csv('/Users/endotakeru/Documents/Interactive-GA-with-GAN/images/lfw.csv')

    def get_target(x):
        if x == "Colin_Powell":
            return 1
        return 0
    df["target"] = df["name"].map(get_target)

    # 整形
    df_1 = df[df["target"] == 0].sample(n=1000)
    df_2 = df[df["target"] == 1]
    df = pd.concat([df_1, df_2])
    df = df.reset_index(drop=True)

    print(df.shape)

    df["fold"] = 0
    Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=241)
    for n, (train_index, val_index) in enumerate(Fold.split(df, df.target)):
        df.loc[val_index, 'fold'] = int(n)

    for fold in range(5):
        if fold > 1:
            continue
        train = df[df.fold != fold]
        val = df[df.fold == fold]
        train_dataset = FaceDataset(train, train_transforms(256))
        val_dataset = FaceDataset(val, valid_transforms(256))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=1,
            pin_memory=False,
            drop_last=True)

        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            drop_last=True)

        loss = torch.nn.BCEWithLogitsLoss()
        model = VGG_MLP().to(device)

        best_score = 0.

        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-5, last_epoch=-1)

        for epoch in range(20):
            train_loss = train_fn(
                epoch, model, loss, optimizer, train_loader, device, scheduler=scheduler)
            with torch.no_grad():
                valid_loss, valid_preds, valid_labels = valid_fn(
                    epoch, model, loss, val_loader, device, scheduler=None)
            score = get_score(valid_labels, valid_preds)
            print(
                f'Epoch {epoch+1} - avg_train_loss: {train_loss:.4f}  avg_val_loss: {valid_loss:.4f}')
            print(f'Epoch {epoch+1} - Accuracy: {score}')
            if score > best_score:
                best_score = score
                print(
                    f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict(),
                            'preds': valid_preds}, f'fold_{fold}_best.pth')
            torch.save(model.state_dict(), f'fold_{fold}_{epoch}.pth')
