import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.data import Mixup
except ImportError:
    Mixup = object
try:
    from timm.utils import accuracy
except ImportError:
    def accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = min(max(topk), output.size(1))
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))
            res = []
            for k in topk:
                kk = min(k, output.size(1))
                correct_k = correct[:kk].reshape(-1).float().sum(0)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
from typing import Iterable, Optional
import utils.misc as misc
import utils.lr_sched as lr_sched
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    multilabel_confusion_matrix, balanced_accuracy_score,
    precision_score, recall_score, cohen_kappa_score,confusion_matrix,matthews_corrcoef
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        writer_dir = getattr(log_writer, 'log_dir', getattr(log_writer, 'logdir', ''))
        print('log_dir: {}'.format(writer_dir))

    for data_iter_step, (samples, masks, targets, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(samples, mask=masks)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def plot_custom_confusion_matrix(y_true, y_pred, save_path,
                                 classes=None, figsize=(8, 6),
                                 cmap='Blues', dpi=600):
    cm_counts = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

    nrows, ncols = cm_counts.shape
    annot_labels = np.empty_like(cm_counts, dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            annot_labels[i, j] = f"{cm_norm[i, j]:.4f}\n({cm_counts[i, j]})"

    if classes is None:
        classes = list(range(ncols))

    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm_norm,
                     annot=annot_labels,
                     fmt='',
                     cmap=cmap,
                     cbar=True,
                     vmin=0, vmax=1,
                     cbar_kws={"ticks": np.arange(0, 1.01, 0.2)},
                     xticklabels=classes,
                     yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

@torch.no_grad()
def evaluate(data_loader, model, device, task, epoch, mode, num_class, update_bn_with_target=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'{mode}:'

    if not os.path.exists(task):
        os.makedirs(task)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    image_paths_list = []

    if update_bn_with_target:
        bn_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
        if not bn_layers:
            print("BN update requested, but the single-branch model contains no BatchNorm layers; skipping.")
        else:
            model.train()
            for batch in data_loader:
                images = batch[0].to(device, non_blocking=True)
                masks = batch[1].to(device, non_blocking=True)
                _ = model(images, mask=masks)
            model.eval()

    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, masks, target, image_paths = batch
        image_paths_list.extend(image_paths)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        true_label = F.one_hot(target.to(torch.int64), num_classes=num_class)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            output = model(images, mask=masks)
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            _, prediction_decode = torch.max(prediction_softmax, 1)
            _, true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

        acc1, _ = accuracy(output, target, topk=(1, 2))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    true_label_onehot_list = np.array(true_label_onehot_list)
    prediction_list = np.array(prediction_list)

    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,
                                                   labels=[i for i in range(num_class)])

    mcc = matthews_corrcoef(true_label_decode_list, prediction_decode_list)
    acc = accuracy_score(true_label_decode_list, prediction_decode_list)
    pred_onehot = np.eye(num_class)[prediction_decode_list]
    f1 = f1_score(true_label_onehot_list, pred_onehot, average='macro', zero_division=0)
    precision = precision_score(true_label_onehot_list, pred_onehot, average='macro', zero_division=0)
    recall = recall_score(true_label_onehot_list, pred_onehot, average='macro', zero_division=0)
    kappa = cohen_kappa_score(true_label_decode_list, prediction_decode_list)
    auc_roc_macro = roc_auc_score(true_label_onehot_list, prediction_list, multi_class='ovr', average='macro')
    auc_pr_macro = average_precision_score(true_label_onehot_list, prediction_list, average='macro')

    df2 = pd.DataFrame({
        'image': image_paths_list,
        'prediction_list': [list(p) for p in prediction_list],
        'true_one_hot_list': [list(t) for t in true_label_onehot_list]
    })
    task_no_dash = os.path.normpath(task)
    df2.to_csv(task + f'{os.path.basename(task_no_dash)}_{mode}_prediction_list.csv', index=None)

    metric_logger.synchronize_between_processes()

    if mode == 'test' or mode == 'val':
        print('Sklearn Metrics - Acc: {:.4f} AUC-roc(macro): {:.4f} AUC-pr(macro): {:.4f} '
              'F1-score: {:.4f} MCC: {:.4f}  PRE: {:.4f} '
              'Recall: {:.4f} Kappa: {:.4f}'.format(
            acc, auc_roc_macro, auc_pr_macro, f1, mcc,precision, recall, kappa))

    class_names = getattr(data_loader.dataset, 'classes', [f'Class_{i}' for i in range(num_class)])
    print("\n=== Per-Class Metrics ===")
    per_class_acc = []
    per_class_auc_roc = []
    per_class_auc_pr = []

    for i in range(num_class):
        true_bin = (true_label_decode_list == i).astype(int)
        pred_bin = (prediction_decode_list == i).astype(int)
        class_acc = accuracy_score(true_bin, pred_bin)
        per_class_acc.append(class_acc)

        try:
            class_auc_roc = roc_auc_score(true_label_onehot_list[:, i], prediction_list[:, i])
        except ValueError:
            class_auc_roc = 0.0

        try:
            class_auc_pr = average_precision_score(true_label_onehot_list[:, i], prediction_list[:, i])
        except ValueError:
            class_auc_pr = 0.0

        per_class_auc_roc.append(class_auc_roc)
        per_class_auc_pr.append(class_auc_pr)

        print(f"Class {i} ({class_names[i]}): "
              f"Acc = {class_acc:.4f}, "
              f"AUC-ROC = {class_auc_roc:.4f}, "
              f"AUC-PR = {class_auc_pr:.4f}")

    results_path = task + f'{os.path.basename(task_no_dash)}_metrics_{mode}.csv'
    with open(results_path, mode='w+', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)

        wf.writerow(['Overall_Acc', 'Overall_Prec',
                     'Overall_Recall', 'Overall_Kappa',
                     'Overall_AUC_ROC', 'Overall_AUC_PR',
                     'Overall_F1', 'Overall_MCC'])

        wf.writerow([acc, precision,
                     recall, kappa,
                     auc_roc_macro, auc_pr_macro, f1, mcc])

        wf.writerow([])
        wf.writerow(['Class', 'Class_Name', 'Accuracy', 'AUC_ROC', 'AUC_PR'])
        for i in range(num_class):
            wf.writerow([i, class_names[i], per_class_acc[i], per_class_auc_roc[i], per_class_auc_pr[i]])

    if mode == 'test':
        save_path = task + 'confusion_matrix_test.jpg'
        plot_custom_confusion_matrix(
            y_true=true_label_decode_list,
            y_pred=prediction_decode_list,
            save_path=save_path,
            classes=class_names
        )

    print(f"\n *** Per-class metrics *** save to: {results_path}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc_roc_macro, auc_pr_macro, acc