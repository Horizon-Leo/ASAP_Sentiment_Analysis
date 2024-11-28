import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from utils import load_data, to_device, set_seed, save_json


class MyModel(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size

        # 注意力机制参数
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Wp = nn.Linear(hidden_size, hidden_size)
        self.Wy = nn.Linear(hidden_size, 18)  # ACSA 的情感极性类别数量
        self.Wg = nn.Linear(hidden_size, 5)  # RP 的评分类别数量
        self.bias = nn.Parameter(torch.zeros(18))
        self.bias_g = nn.Parameter(torch.zeros(5))

    def forward(self, input_ids, attention_mask=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        bert_output = self.dropout(bert_output)

        # 注意力得分的中间表示
        Ma = torch.tanh(self.Wa(bert_output))
        alpha = torch.softmax(Ma, dim=1)
        r = torch.tanh(self.Wp(torch.sum(bert_output * alpha, dim=1)))

        # ACSA 输出
        logits_acsa = torch.softmax(self.Wy(r) + self.bias, dim=1)

        # RP 输出
        logits_rp = torch.softmax(self.Wg(r) + self.bias_g, dim=1)

        return logits_acsa, logits_rp


def compute_loss(logits_acsa, targets_y, logits_rp, targets_g, acsa_weight=0.7, rp_weight=0.3):
    loss_y = nn.CrossEntropyLoss()(logits_acsa, targets_y)
    # 使用 CrossEntropyLoss 计算 RP 的损失
    loss_g = nn.CrossEntropyLoss()(logits_rp, targets_g)
    return loss_y * acsa_weight + loss_g * rp_weight


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, writer):
    model.train()
    train_history = []
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch:02d}")
    for batch_idx, (inputs_tokenized, targets_y, targets_g) in enumerate(progress_bar):
        optimizer.zero_grad()

        inputs_tokenized = {k: v.to(device) for k, v in inputs_tokenized.items()}
        targets_y = targets_y.to(device)
        targets_g = targets_g.to(device)

        logits_acsa, logits_rp = model(input_ids=inputs_tokenized['input_ids'],
                                       attention_mask=inputs_tokenized['attention_mask'])

        loss = compute_loss(logits_acsa, targets_y, logits_rp, targets_g)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_history.append(loss.item())
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 计算并记录 ACSA 的 F1 和准确率
        y_pred = logits_acsa.argmax(dim=1)
        f1 = f1_score(targets_y.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
        accuracy = accuracy_score(targets_y.cpu().numpy(), y_pred.cpu().numpy())

        # 计算并记录 RP 的准确率和 F1
        g_pred = logits_rp.argmax(dim=1)
        rp_f1 = f1_score(targets_g.cpu().numpy(), g_pred.cpu().numpy(), average='macro')
        rp_accuracy = accuracy_score(targets_g.cpu().numpy(), g_pred.cpu().numpy())

        writer.add_scalar('F1/train', f1, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Accuracy/train', accuracy, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('RP_F1/train', rp_f1, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('RP_Accuracy/train', rp_accuracy, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

    return train_history


def val_epoch(model, val_loader, device, writer, epoch, phase='val'):
    model.eval()
    total_loss = 0.0
    total_f1 = 0.0
    total_accuracy = 0.0
    total_rp_f1 = 0.0
    total_rp_accuracy = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation {phase}"):
            inputs_tokenized, targets_y, targets_g = batch
            inputs_tokenized = {k: v.to(device) for k, v in inputs_tokenized.items()}
            targets_y = targets_y.to(device)
            targets_g = targets_g.to(device)

            logits_acsa, logits_rp = model(input_ids=inputs_tokenized['input_ids'],
                                           attention_mask=inputs_tokenized['attention_mask'])

            loss = compute_loss(logits_acsa, targets_y, logits_rp, targets_g)
            total_loss += loss.item()

            # 计算 ACSA 评估指标
            y_pred = logits_acsa.argmax(dim=1)
            f1 = f1_score(targets_y.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
            accuracy = accuracy_score(targets_y.cpu().numpy(), y_pred.cpu().numpy())

            # 计算 RP 评估指标
            g_pred = logits_rp.argmax(dim=1)
            rp_f1 = f1_score(targets_g.cpu().numpy(), g_pred.cpu().numpy(), average='macro')
            rp_accuracy = accuracy_score(targets_g.cpu().numpy(), g_pred.cpu().numpy())

            total_f1 += f1
            total_accuracy += accuracy
            total_rp_f1 += rp_f1
            total_rp_accuracy += rp_accuracy

    avg_loss = total_loss / len(val_loader)
    avg_f1 = total_f1 / len(val_loader)
    avg_accuracy = total_accuracy / len(val_loader)
    avg_rp_f1 = total_rp_f1 / len(val_loader)
    avg_rp_accuracy = total_rp_accuracy / len(val_loader)

    # Log validation metrics to TensorBoard
    writer.add_scalar(f'Loss/{phase}', avg_loss, epoch)
    writer.add_scalar(f'F1/{phase}', avg_f1, epoch)
    writer.add_scalar(f'Accuracy/{phase}', avg_accuracy, epoch)
    writer.add_scalar(f'RP_F1/{phase}', avg_rp_f1, epoch)
    writer.add_scalar(f'RP_Accuracy/{phase}', avg_rp_accuracy, epoch)

    return avg_loss


def train_and_validation(model, train_loader, val_loader, test_loader, optimizer, scheduler, num_epochs, device):
    writer = SummaryWriter(log_dir='runs/experiment_2')
    best_val_f1 = 0
    train_history, metrics = [], []
    for epoch in range(num_epochs):
        history = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, writer)
        val_loss = val_epoch(model, val_loader, device, writer, epoch, phase='val')
        test_loss = val_epoch(model, test_loader, device, writer, epoch, phase='test')

        print('Val: ', val_loss)
        print('Test: ', test_loss)

        train_history.extend(history)
        metrics.append((val_loss, test_loss))

        save_json(
            {
                'train_history': train_history,
                'metrics': metrics
            },
            os.path.join('metrics.json')
        )

        if val_loss < best_val_f1:
            print(f'当前验证集最低损失: {val_loss}')
            best_val_f1 = val_loss
            torch.save({
                'epoch': epoch,
                'loss': best_val_f1,
                'model_state_dict': model.state_dict(),
            }, '../best_model.pt')

    writer.close()


def main(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.roberta_path)
    bert_model = AutoModel.from_pretrained(args.roberta_path)

    train_dataloader, val_dataloader, test_dataloader = load_data(args.batch_size, tokenizer)

    my_model = MyModel(bert_model)
    my_model.to(args.device)

    optimizer = AdamW(my_model.parameters(), lr=args.lr)
    num_iter_per_epoch = len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_iter_per_epoch,
        num_training_steps=num_iter_per_epoch * args.num_epochs
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
    parser.add_argument('--roberta_path', type=str, default=r'C:\Code\Sentiment_analysis-local\detail')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(args)