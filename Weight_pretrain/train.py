import numpy as np
import os
import torch
from pandas.core.frame import DataFrame
from torchvision import transforms, datasets
from data import ImageFolderCustom
from utils import train_one_epoch, evaluate, cosine_scheduler
from network_multi_label import MultiAV4
global log
log = []
import time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    start_time = time.time()
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(start_time))

    image_path ='/home/hyding/huaiyuan_code/paper_test/test/'
    csv_path = f'/home/hyding/huaiyuan_code/paper_test/test/outweights/{start_time}.csv'

    batch_size = 256
    freeze_layers = False
    learning_rate = 5e-2  # pretrain
    weight_decay = 0
    step_size = 50
    gamma = 0.5
    early_stop_step = 20
    epochs = 300
    best_acc = 0.5
    patch_size = 256
    print('patch_size =', patch_size)
    print('batch_size =', batch_size )

    data_transform = {
        "train": transforms.Compose([transforms.Resize((patch_size,patch_size)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.RandomRotation(180),
                                     transforms.RandomGrayscale(p=0.2),
                                     transforms.ColorJitter(0.1, 0.1, 0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "test": transforms.Compose([transforms.Resize((patch_size,patch_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = ImageFolderCustom(os.path.join(image_path, "training"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 10])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               drop_last=True
                                            )
    
    validate_dataset =ImageFolderCustom(os.path.join(image_path, "test"),
                                         transform=data_transform["test"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw,
                                                  drop_last=True)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    model = MultiAV4(num_classes=2, pretrain=True)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    print("\n========== Model Init Info ==========")

    if hasattr(model, "sn_unet"):
        print("Backbone: ConvNeXtV2-Tiny")

    if True:
        print("Using ImageNet Pretrained Weights (ConvNeXtV1_Tiny)")
    else:
        print("Training from Scratch (No Pretrained Weights)")

    print("Model: MultiAV_ROP")
    print("=======================================\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-3
    )
    it = train_num//batch_size
    lr_sc = cosine_scheduler(5e-4, 1e-6, 151, it, 0)
    total_batch = 0
    last_decrease = 0
    min_loss = 1000
    flag = False
    val_best_loss = np.inf
    train_best_loss = np.inf
    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                start_steps=epoch * it,
                                                lr_sc =lr_sc,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch, )

        val_loss,val_acc = evaluate(model=model,
                                     data_loader=validate_loader,
                                     device=device,
                                     epoch=epoch)

        if (epoch + 1) % 1 == 0:
            checkpoint_path = fr"/home/hyding/huaiyuan_code/paper_test/test/outweights/checkpoint_epoch_{epoch + 1}_{start_time}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")

        if val_acc > best_acc:
            print(val_acc,best_acc)
            best_acc = val_acc
            torch.save(model.state_dict(), fr"/home/hyding/huaiyuan_code/paper_test/test/outweights/train_best_acc_model_rop_soft_{start_time}.pth")

        if val_loss < min_loss:
            min_loss = val_loss
            last_decrease = total_batch
            print((min_loss, last_decrease))
            torch.save(model.state_dict(), fr"/home/hyding/huaiyuan_code/paper_test/test/outweights/train_best_loss_model_rop_soft_{start_time}.pth")
        total_batch += 1

        if total_batch - last_decrease > early_stop_step:
            print("No optimization for a long time, auto-stopping...")
            flag = True
            break
        log.append([epoch, train_loss, val_loss, train_acc, val_acc])
    print('Finished Training')
    data = DataFrame(data=log, columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    data.to_csv(csv_path)

if __name__ == '__main__':
    main()
