import numpy as np
import torch
from tqdm import tqdm

from torch.nn import functional as F
from utils import save_checkpoint


def train(cfg, device, model, optimizer, criterion, train_loader, val_loader, mproc):

    for epoch in tqdm(range(cfg.hypers['num_epochs'])):
        print(f'Epoch {epoch + 1}')

        pbar = tqdm(
            train_loader,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            colour='green')

        # Best values placeholder
        best = dict()
        best['val_loss'] = np.inf

        model.train()

        for idx, batch in enumerate(pbar):
            # Get the inputs
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Metric Processor
            preds_np = outputs.detach().cpu().numpy()
            gts_np = labels.cpu().numpy()
            mproc.add_loss(loss.item(), phase='train')
            mproc.add_accuracy(preds_np, gts_np, phase='train')
            mproc.add_f1(preds_np, gts_np, phase='train')
            mproc.add_precision(preds_np, gts_np, phase='train')
            mproc.add_recall(preds_np, gts_np, phase='train')

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        else:
            model.eval()

            with torch.no_grad():

                for idx, batch in enumerate(val_loader):
                    # Get the inputs
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Metric Processor for current batch
                    preds_np = outputs.detach().cpu().numpy()
                    gts_np = labels.cpu().numpy()
                    mproc.add_loss(loss.item(), phase='val')
                    mproc.add_accuracy(preds_np, gts_np, phase='val')
                    mproc.add_f1(preds_np, gts_np, phase='val')
                    mproc.add_precision(preds_np, gts_np, phase='val')
                    mproc.add_recall(preds_np, gts_np, phase='val')

        # Metric Processor for epoch
        mproc.get_epoch_loss()
        print(
            f'Total for epoch {epoch}:\n'
            f'Train loss: {"{:.4f}".format(mproc.train["loss"].total[-1])}\n'
            f'Val loss: {"{:.4f}".format(mproc.val["loss"].total[-1])}\n'
            f'Train accuracy: {"{:.4f}".format(mproc.train["accuracy"].total[-1])}\n'
            f'Val accuracy: {"{:.4f}".format(mproc.val["accuracy"].total[-1])}\n'
            f'Train F1: {"{:.4f}".format(mproc.train["f1"].total[-1])}\n'
            f'Val F1: {"{:.4f}".format(mproc.val["f1"].total[-1])}\n'
            f'Train Precision: {"{:.4f}".format(mproc.train["precision"].total[-1])}\n'
            f'Val Precision: {"{:.4f}".format(mproc.val["precision"].total[-1])}\n'
            f'Train Recall: {"{:.4f}".format(mproc.train["recall"].total[-1])}\n'
            f'Val Recall: {"{:.4f}".format(mproc.val["recall"].total[-1])}\n'
        )

        # Save model if it is got better
        print('CHECK:', mproc.val['loss'].total[-1], mproc.best_val_loss)
        if mproc.val['loss'].total[-1] < mproc.best_val_loss:
            best = dict()
            best['train_loss'] = mproc.train['loss'].total[-1]
            best['val_loss'] = mproc.train['loss'].total[-1]
            best['model_state_dict'] = model.state_dict()
            best['optimizer_state_dict'] = optimizer.state_dict()
            best['epoch'] = epoch
            save_checkpoint(cfg, best)


    print('Finished Training')

    mproc.plot_metrics()