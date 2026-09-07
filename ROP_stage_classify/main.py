import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

try:
    import timm
    from timm.models.layers import trunc_normal_
    from timm.data.mixup import Mixup
    from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
except ImportError:
    timm = None
    from torch.nn.init import trunc_normal_

    class Mixup:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Mixup requires timm; install requirements.txt or keep --mixup/--cutmix at 0.")

    class SoftTargetCrossEntropy(torch.nn.Module):
        def forward(self, x, target):
            return torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1).mean()

    class LabelSmoothingCrossEntropy(torch.nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing

        def forward(self, x, target):
            return F.cross_entropy(x, target, label_smoothing=self.smoothing)

import utils.lr_decay as lrd
import utils.misc as misc
from dataset import build_dataset
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from model import Multi_ROP, load_checkpoint
from collections import defaultdict
from engine import train_one_epoch, evaluate
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from torch.utils.data import Sampler
from operator import itemgetter

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, images, mask=None, class_idx=None):
        self.model.eval()
        output = self.model(images, mask=mask)
        if class_idx is None:
            class_idx = output.argmax(dim=1)

        batch_cams = []
        for i in range(images.size(0)):
            self.model.zero_grad()
            output[i, class_idx[i]].backward(retain_graph=True)
            if self.activations is None or self.gradients is None:
                batch_cams.append(np.zeros((images.shape[2], images.shape[3])))
                continue
            act = self.activations[i]
            grad = self.gradients[i]
            weights = grad.mean(dim=(1, 2))
            cam = (weights[:, None, None] * act).sum(dim=0)
            cam = F.relu(cam)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            batch_cams.append(cam.cpu().numpy())
        return batch_cams, class_idx.detach().cpu().numpy()


def visualize_gradcam_per_class(
        imgs, cams, pred_classes, true_classes,
        class_names, idx=0, save_path=None
):

    img = imgs[idx]
    if img.requires_grad:
        img = img.detach()

    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = std * img + mean
    img = np.clip(img, 0.0, 1.0)
    cam = cams[idx].astype(np.float32)
    cam = gaussian_filter(cam, sigma=1.0)

    cam_resized = cv2.resize(
        cam,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    cam_resized = np.maximum(cam_resized, 0)
    cam_resized -= cam_resized.min()
    cam_max = cam_resized.max()

    if cam_max > 1e-8:
        cam_resized /= cam_max
    else:
        cam_resized[:] = 0.0

    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    heatmap_uint8 = np.round(cam_resized * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0

    overlay = 0.35 * img + 0.65 * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    true_label = class_names[int(true_classes[idx])]
    pred_label = class_names[int(pred_classes[idx])]
    title = f"True: {true_label} | Pred: {pred_label}"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(
        title,
        fontsize=14,
        color="red"
        if int(pred_classes[idx]) != int(true_classes[idx])
        else "green"
    )
    axes[1].axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
            facecolor="white"
        )

    plt.close(fig)

class WeightedLabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, smoothing=0.1, weight=None):
        super(WeightedLabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.weight = weight

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.weight is not None:
            loss *= self.weight[target]

        return loss.mean()


def saferound(x, digits=0):
    rounded = np.round(x - 0.5e-10, decimals=digits).astype(int)
    error = x - rounded
    if not np.isclose(error.sum(), 0):
        diff = int(round(error.sum()))
        rounded[-1] += diff
    return rounded


class WeightedBalanceClassSampler(Sampler):
    def __init__(
            self, labels, weight, length,
    ):
        super().__init__(labels)

        labels = np.array(labels).astype(np.int32)

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }
        weight = np.array(weight)
        weight = weight / weight.sum()

        samples_per_class = weight * length

        samples_per_class = np.array(saferound(samples_per_class, 0)).astype(np.int32)

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = length

    def __iter__(self):
        np.random.seed(self.seed + self.epoch if hasattr(self, 'epoch') else 42)

        indices = []
        for key in sorted(self.lbl2idx):
            replace_flag = self.samples_per_class[key] > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class[key], replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        return self.length


class DatasetFromSampler(torch.utils.data.Dataset):
    def __init__(self, sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self):
        return len(self.sampler)


class DistributedSamplerWrapper(torch.utils.data.DistributedSampler):
    def __init__(
            self,
            sampler=None,
            num_replicas=1,
            rank=0,
            shuffle=True,
    ):
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def set_requires_grad(nets, requires_grad=True):
    for name, param in nets.named_parameters():
        if 'sn_unet' in name or 'branch' in name:
            param.requires_grad = requires_grad
    for name, param in nets.named_parameters():
        print(f'{name}: {param.requires_grad}')
    return nets


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.3, metavar='PCT',
                        help='Drop path rate (default: 0.3)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.9,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=5e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--color_jitter', type=float, default=0.2, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--finetune',
                        default='./Weight_pretrain/out_weight_paper/checkpoint_epoch_9_2026-08-21-16-21.pth',
                        type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--cls_weight', action='store_true',
                        help='Perform cls_weight only')
    parser.add_argument('--data_path',
                        default='./Data_classify/',
                        type=str,
                        help='dataset path')
    # ==== 新增: mask 数据路径 =======
    parser.add_argument('--mask_data_path',
                        default='./Mask_generation/mask_out/',
                        type=str,
                        help='dataset path for UNet predicted glare masks')

    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--visualize_gradcam', action='store_true', default=False,
                        help='Enable GradCAM visualization during validation. Default: False.')
    parser.add_argument('--gradcam_freq', type=int, default=10,
                        help='Generate GradCAM every N epochs (default: 1).')

    return parser

def visualize_all_test_gradcam(args, model, data_loader_test, device, class_names, save_dir):
    """Generate Grad-CAM for every test fused image."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    gradcam = GradCAM(model, model.extra)
    pbar = tqdm(total=len(data_loader_test), desc="GradCAM 生成中", unit="batch")

    for images, masks, targets, image_names in data_loader_test:
        images = images.to(device)
        masks = masks.to(device)
        targets = targets.to(device)
        images.requires_grad_(True)
        cams, pred_classes = gradcam.generate(images, mask=masks)

        for i in range(len(images)):
            base_name = os.path.splitext(image_names[i])[0]
            true_label = class_names[targets[i].item()]
            pred_label = class_names[pred_classes[i]]
            save_path = os.path.join(
                save_dir, f"{base_name}_true_{true_label}_pred_{pred_label}.png"
            )
            visualize_gradcam_per_class(
                images.detach(), cams, pred_classes, targets.detach().cpu().numpy(),
                class_names, idx=i, save_path=save_path
            )
        pbar.update(1)
    pbar.close()


def main(args):
    import random
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    print(f'seed is: {seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)
    dataset_test = build_dataset(is_train='test', args=args)

    if args.cls_weight:
        class_counts = defaultdict(int)
        label_list = []
        for _, _, _, label in dataset_train:
            class_counts[label] += 1
            label_list.append(label)
        total_samples = len(dataset_train)

        class_proportions = {cls: count / total_samples for cls, count in class_counts.items()}

        class_weights = {cls: 1.0 / prop for cls, prop in class_proportions.items()}
        max_weight = max(class_weights.values())
        min_weight = min(class_weights.values())
        max_allowed_weight = min_weight * 10

        max_allowed_weight = min_weight * 10
        adjusted_class_weights = {cls: min(weight, max_allowed_weight) for cls, weight in class_weights.items()}
        total_adjusted_weights = sum(adjusted_class_weights.values())
        final_class_weights = {cls: weight / total_adjusted_weights for cls, weight in adjusted_class_weights.items()}

        weights_tensor = torch.tensor([final_class_weights[cls] for cls in range(len(class_counts))],
                                      dtype=torch.float32)

        imblanceSampler = WeightedBalanceClassSampler(labels=label_list, weight=weights_tensor, length=total_samples)

        print(f'========================weight: {weights_tensor}==============')

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.cls_weight:
            sampler_train = DistributedSamplerWrapper(imblanceSampler, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = None
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    torch.manual_seed(seed)
    model = Multi_ROP(num_classes=args.nb_classes, drop_path=args.drop_path,pretrain=False)

    if args.finetune and os.path.exists(args.finetune):
        print(f"Loading finetune checkpoint from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)

        new_state_dict = {}
        loaded_cnt = 0

        for k, v in state_dict.items():
            key = k[len('module.'):] if k.startswith('module.') else k

            if key.startswith('sn_unet.'):
                body = key[len('sn_unet.'):]
                new_state_dict[f'backbone.{body}'] = v
                loaded_cnt += 1
            elif key.startswith('branch_orig.'):
                body = key[len('branch_orig.'):]
                new_state_dict[f'backbone.{body}'] = v
                loaded_cnt += 1
            elif key.startswith('extra_orig.'):
                body = key[len('extra_orig.'):]
                new_state_dict[f'extra.{body}'] = v
                loaded_cnt += 1


        print(f"处理 {loaded_cnt} 个 backbone 权重")
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys (extra/head): {len(msg.missing_keys)}")
        print(f"Unexpected keys: {len(msg.unexpected_keys)}")

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    bb_keys = ('backbone',)
    backbone_params = [p for n, p in model_without_ddp.named_parameters() if n.startswith(bb_keys)]
    new_params = [p for n, p in model_without_ddp.named_parameters() if not n.startswith(bb_keys)]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': new_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))



    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:

        test_stats, auc_roc, _, _ = evaluate(data_loader_test, model, device, args.task, epoch=0, mode='test', num_class=args.nb_classes)

        if misc.is_main_process():
            model_without_ddp.eval()

        #GradCAM
        if args.visualize_gradcam:
            cam_save_dir = os.path.join(args.task, "gradcam_test_all")
            visualize_all_test_gradcam(
                args, model_without_ddp, data_loader_test,
                device, dataset_test.classes, cam_save_dir
            )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.5
    max_auc = 0.0
    max_pr = 0.0
    last_pr_auc = 0.0
    last_acc_auc = 0.0
    last_pr = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        val_stats, val_auc_roc, val_auc_pr, val_acc = evaluate(data_loader_val, model, device, args.task, epoch, mode='val', num_class=args.nb_classes)

        if misc.is_main_process():

            class_names = data_loader_val.dataset.classes
            num_classes = len(class_names)
            selected_images = []
            selected_masks = []
            selected_true = []
            seen = set()

            for images_batch, masks_batch, targets_batch, *_ in data_loader_val:
                for j in range(len(targets_batch)):
                    label = targets_batch[j].item()
                    if label not in seen:
                        seen.add(label)
                        selected_images.append(images_batch[j])
                        selected_masks.append(masks_batch[j])
                        selected_true.append(label)
                        if len(seen) == num_classes:
                            break
                if len(seen) == num_classes:
                    break

            if args.visualize_gradcam and selected_images and epoch % args.gradcam_freq == 0:
                cam_dir = r'./gradcam/'
                os.makedirs(cam_dir, exist_ok=True)

                _model = model_without_ddp
                _model.eval()
                gradcam = GradCAM(_model, _model.extra)
                selected_images_tensor = torch.stack(selected_images).to(device)
                selected_masks_tensor = torch.stack(selected_masks).to(device)
                selected_images_tensor.requires_grad_(True)
                cams, pred_classes = gradcam.generate(
                    selected_images_tensor,
                    mask=selected_masks_tensor,
                )

                true_classes = np.array(selected_true)

                for i in range(len(selected_images)):
                    true_name = class_names[true_classes[i]]
                    pred_name = class_names[pred_classes[i]]
                    save_path = os.path.join(
                        cam_dir,
                        f'epoch{epoch:03d}_fused_class{true_name}_true{true_name}_pred{pred_name}.png'
                    )
                    visualize_gradcam_per_class(
                        selected_images_tensor.detach(), cams, pred_classes, true_classes,
                        class_names, idx=i, save_path=save_path
                    )

                print(f" *** GradCAM *** save to: {cam_dir}")
                _model.train()

        if (last_pr < val_auc_pr and max_auc == val_auc_roc) or (max_auc < val_auc_roc):
            max_auc = max(val_auc_roc, max_auc)
            last_pr = val_auc_pr

            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch='best_auc')
                test_stats, auc_roc, auc_pr, acc = evaluate(data_loader_test, model, device, args.task, epoch,
                                                            mode='test', num_class=args.nb_classes)

                print(f'Max AUC: {max_auc:.2f}%   Test: {auc_roc:.2f}%')

        if epoch == (args.epochs - 1):
            test_stats, auc_roc, _, _ = evaluate(data_loader_test, model, device, args.task, epoch, mode='test',
                                                 num_class=args.nb_classes)
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        if log_writer is not None:
            log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
            log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    with open(os.path.join(args.output_dir, "time.txt"), mode="w", encoding="utf-8") as f:
        f.write(f'Training time {total_time_str}')
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

