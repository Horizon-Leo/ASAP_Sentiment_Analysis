import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch import nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel

from utils import load_data, to_device, get_lr_scheduler, set_seed, save_json


class MyModel(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size
        self.fcs = nn.ModuleList(
            [nn.Linear(hidden_size, 5)] + [nn.Linear(hidden_size, 4) for _ in range(18)]
        )

    def forward(self, x):
        bert_cls = self.bert(**x).last_hidden_state[:, 0, :]
        bert_cls = self.dropout(bert_cls)
        logits_rating = self.fcs[0](bert_cls)
        logits_cats = torch.cat([fc(bert_cls) for fc in self.fcs[1:]], dim=0)
        return logits_rating, logits_cats

    def _init_weights(self):
        for p in self.fcs.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    train_history = []
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch:02d}")
    for batch in progress_bar:
        optimizer.zero_grad()
        inputs_tokenized, targets_rating, targets_cats = to_device(batch[:-1], device)  # targets: (batch, 19)
        logits_rating, logits_cats = model(inputs_tokenized)

        loss_rating = nn.CrossEntropyLoss()(logits_rating, targets_rating)
        loss_cats = nn.CrossEntropyLoss()(logits_cats, targets_cats)
        loss = loss_rating + loss_cats

        loss.backward()
        optimizer.step()
        scheduler.step()
        train_history.append(loss.item())
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    return train_history


def val_epoch(model, val_loader, device):
    model.eval()
    total_loss, pred_list, tar_list = 0.0, [], []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs_tokenized, targets_rating, targets_cats = to_device(batch[:-1], device)  # targets: (batch, 19)
            logits_rating, logits_cats = model(inputs_tokenized)

            loss_rating = nn.CrossEntropyLoss()(logits_rating, targets_rating)
            loss_cats = nn.CrossEntropyLoss()(logits_cats, targets_cats)
            loss = loss_rating + loss_cats

            pred_list.append(logits_rating.argmax(dim=-1).cpu().tolist() + logits_cats.argmax(dim=-1).cpu().tolist())
            tar_list.append(targets_rating.cpu().tolist() + targets_cats.cpu().tolist())

            total_loss += loss.item()

    all_f1, all_acc = [], []
    for i in range(19):
        ps = [pred[i] for pred in pred_list]
        ts = [tar[i] for tar in tar_list]

        f1_macro = f1_score(ts, ps, average='macro')
        all_f1.append(f1_macro)

        acc = accuracy_score(ts, ps)
        all_acc.append(acc)

    f1_mean = sum(all_f1) / len(all_f1)
    acc_mean = sum(all_acc) / len(all_acc)

    return total_loss / len(val_loader), f1_mean, acc_mean


def train_and_validation(model, train_loader, val_loader, test_loader, optimizer, scheduler,
                         num_epochs, device):
    best_val_f1 = 0
    train_history, metrics = [], []
    for epoch in range(num_epochs):
        history = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss, val_f1_mean, val_acc_mean = val_epoch(model, val_loader, device)
        test_loss, test_f1_mean, test_acc_mean = val_epoch(model, test_loader, device)
        print('Val: ', val_loss, val_f1_mean, val_acc_mean)
        print('Test: ', test_loss, test_f1_mean, test_acc_mean)

        train_history.extend(history)
        metrics.append((val_loss, val_f1_mean, val_acc_mean, test_loss, test_f1_mean, test_acc_mean))

        save_json(
            {
                'train_history': train_history,
                'metrics': metrics
            },
            os.path.join('metrics.json')
        )

        if val_f1_mean > best_val_f1:
            print(f'当前验证集最高F1: {val_f1_mean}')
            best_val_f1 = val_f1_mean
            torch.save({
                'epoch': epoch,
                'f1': best_val_f1,
                'model_state_dict': model.state_dict(),
            }, 'best_model.pt')


def main(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.roberta_path)
    bert_model = AutoModel.from_pretrained(args.roberta_path)

    train_dataloader, val_dataloader, test_dataloader = load_data(args.batch_size, tokenizer)

    print('#' * 20)
    print('device: ', args.device)
    print('#' * 20)

    my_model = MyModel(bert_model)
    my_model.to(args.device)

    optimizer = Adam(my_model.parameters(), lr=args.lr)
    num_iter_per_epoch = len(train_dataloader)
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=num_iter_per_epoch,
        total_steps=num_iter_per_epoch * args.num_epochs,
        decrease_mode=args.lr_decrease_model
    )

    train_and_validation(
        my_model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        args.num_epochs,
        args.device
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roberta_path', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_decrease_model', type=str, default='cosin', choices=['cosin', 'linear', 'const'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(args)

# nohup python train.py > train.log 2>&1 &
