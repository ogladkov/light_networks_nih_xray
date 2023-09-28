import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch


def save_checkpoint(cfg, best):
    save_path = f"{cfg.hypers['checkpoint_save_path']}_{cfg.hypers['model_name'].split('/')[-1]}_{best['epoch']}_{'%.4f' % best['val_loss']}.pt"
    torch.save({
        'epoch': best['epoch'],
        'model_state_dict': best['model_state_dict'],
        'optimizer_state_dict': best['optimizer_state_dict'],
        'loss': best['val_loss'],
    }, save_path)


class Metric:

    def __init__(self, name):
        self.name = name
        self.total = []
        self.current = []

    def __repr__(self):
        return self.name

class MetricProcessor:

    def __init__(self, metrics: list):
        self.metrics = metrics
        self.train = {m: Metric(m) for m in metrics}
        self.val = {m: Metric(m) for m in metrics}
        self.best_val_loss = np.inf

    def add_loss(self, loss_value, phase='train'):

        # Check best val loss before adding
        if len(self.val['loss'].total) > 0:
            self.best_val_loss = np.min(self.val['loss'].total)

        if phase == 'train':
            self.train['loss'].current.append(loss_value)

        elif phase == 'val':
            self.val['loss'].current.append(loss_value)

    def add_accuracy(self, preds, gts, phase='train'):
        preds = np.argmax(preds, axis=1)
        accuracy = accuracy_score(gts, preds.astype(int))

        if phase == 'train':
            self.train['accuracy'].current.append(accuracy)

        elif phase == 'val':
            self.val['accuracy'].current.append(accuracy)

    def add_f1(self, preds, gts, phase='train'):
        preds = np.argmax(preds, axis=1)
        f1 = f1_score(gts, preds.astype(int), average='weighted')

        if phase == 'train':
            self.train['f1'].current.append(f1)

        elif phase == 'val':
            self.val['f1'].current.append(f1)

    def add_precision(self, preds, gts, phase='train', threshold=0.5):
        preds = np.argmax(preds, axis=1)
        precision = precision_score(gts, preds.astype(int), average='weighted')

        if phase == 'train':
            self.train['precision'].current.append(precision)

        elif phase == 'val':
            self.val['precision'].current.append(precision)

    def add_recall(self, preds, gts, phase='train', threshold=0.5):
        preds = np.argmax(preds, axis=1)
        recall = recall_score(gts, preds.astype(int), average='weighted')

        if phase == 'train':
            self.train['recall'].current.append(recall)

        elif phase == 'val':
            self.val['recall'].current.append(recall)

    def get_epoch_loss(self):

        for metric in self.metrics:
            self.train[metric].total.append(np.mean(self.train[metric].current))
            self.val[metric].total.append(np.mean(self.val[metric].current))
            self.train[metric].current = []

    def plot_metrics(self):
        fig, axs = plt.subplots(3, 2, figsize=(16, 16))

        axs[0, 0].plot(range(1, len(self.train['loss'].total) + 1), self.train['loss'].total, label='Train')
        axs[0, 0].plot(range(1, len(self.val['loss'].total) + 1), self.val['loss'].total, label='Val')
        axs[0, 0].set_title('Loss')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        axs[0, 1].axis('off')

        axs[1, 0].plot(range(1, len(self.train['accuracy'].total) + 1), self.train['accuracy'].total, label='Train')
        axs[1, 0].plot(range(1, len(self.val['accuracy'].total) + 1), self.val['accuracy'].total, label='Val')
        axs[1, 0].set_title('Accuracy')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        axs[1, 1].plot(range(1, len(self.train['f1'].total) + 1), self.train['f1'].total, label='Train')
        axs[1, 1].plot(range(1, len(self.val['f1'].total) + 1), self.val['f1'].total, label='Val')
        axs[1, 1].set_title('F1')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        axs[2, 0].plot(range(1, len(self.train['precision'].total) + 1), self.train['precision'].total, label='Train')
        axs[2, 0].plot(range(1, len(self.val['precision'].total) + 1), self.val['precision'].total, label='Val')
        axs[2, 0].set_title('Precision')
        axs[2, 0].grid(True)
        axs[2, 0].legend()

        axs[2, 1].plot(range(1, len(self.train['recall'].total) + 1), self.train['recall'].total, label='Train')
        axs[2, 1].plot(range(1, len(self.val['recall'].total) + 1), self.val['recall'].total, label='Val')
        axs[2, 1].set_title('Recall')
        axs[2, 1].grid(True)
        axs[2, 1].legend()

        plt.tight_layout()

        plt.show()