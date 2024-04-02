import os
from pathlib import Path
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, models

import hfai
import hfai.distributed as dist
hfai.nn.functional.set_replace_torch()
import time


def train(dataloader, model, criterion, optimizer, epoch, local_rank, start_step, best_acc):
    model.train()
    ts = time.time()
    for step, batch in enumerate(dataloader):
        step += start_step

        samples, labels = [x.cuda(non_blocking=True) for x in batch]
        outputs = model(samples)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if local_rank == 0 and step % 20 == 0:
            cost_time = time.time() - ts
            ts = time.time()
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Cost {cost_time}s, {cost_time/20.0} seconds/step', flush=True)
        # save checkpiont
        model.try_save(epoch, step + 1, others=best_acc)


def validate(dataloader, model, criterion, epoch, local_rank):
    loss, correct1, correct5, total = torch.zeros(4).cuda()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            samples, labels = [x.cuda(non_blocking=True) for x in batch]
            outputs = model(samples)
            loss += criterion(outputs, labels)
            _, preds = outputs.topk(5, -1, True, True)
            correct1 += torch.eq(preds[:, :1], labels.unsqueeze(1)).sum()
            correct5 += torch.eq(preds, labels.unsqueeze(1)).sum()
            total += samples.size(0)

    for x in [loss, correct1, correct5, total]:
        dist.reduce(x, 0)

    if local_rank == 0:
        loss_val = loss.item() / dist.get_world_size() / len(dataloader)
        acc1 = 100 * correct1.item() / total.item()
        acc5 = 100 * correct5.item() / total.item()
        print(f'Epoch: {epoch}, Loss: {loss_val}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%', flush=True)

    return correct1.item() / total.item()


def main(local_rank):
    # hyper parameters
    epochs = 100
    batch_size = 100
    lr = 0.015
    # save_path = 'output/resnet'
    save_path = os.environ.get('CHECKPOINT_DIR','output/resnet')
    Path(save_path).mkdir(exist_ok=True, parents=True)

    # multi-node communicate
    ip = os.environ['MASTER_ADDR']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    gpus = torch.cuda.device_count()
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://{ip}:{port}',
                            world_size=hosts * gpus,
                            rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    # Model
    model = models.resnet50().cuda()
    model = hfai.nn.to_hfai(model)
    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])
    # Data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = hfai.datasets.ImageNet('train', transform=train_transform)
    train_datasampler = DistributedSampler(train_dataset)
    train_dataloader = train_dataset.loader(batch_size, sampler=train_datasampler, num_workers=4, pin_memory=True)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_dataset = hfai.datasets.ImageNet('val', transform=val_transform)
    val_datasampler = DistributedSampler(val_dataset)
    val_dataloader = val_dataset.loader(batch_size, sampler=val_datasampler, num_workers=4, pin_memory=True)
    # loss
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    ckpt_path = os.path.join(save_path, 'latest.pt')
    
    start_epoch, start_step, best_acc = hfai.checkpoint.init(model, optimizer, scheduler=scheduler, ckpt_path=ckpt_path)
    best_acc = best_acc or 0

    # Training & Validate
    for epoch in range(start_epoch, epochs):
        # resume from epoch and step
        train_datasampler.set_epoch(epoch)
        train_dataloader.set_step(start_step)

        train(train_dataloader, model, criterion, optimizer, epoch, local_rank, start_step, best_acc)
        start_step = 0  # reset
        scheduler.step()
        acc = validate(val_dataloader, model, criterion, epoch, local_rank)
        # save
        if rank == 0 and local_rank == 0:
            if epoch % 10 == 0:
                torch.save(model.module.state_dict(),
                           os.path.join(save_path, f'ckpt_{epoch}.pt'))
            if acc > best_acc:
                best_acc = acc
                print(f'New Best Acc: {100*acc:.2f}%!')
                torch.save(model.module.state_dict(),
                           os.path.join(save_path, 'best.pt'))


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus)
