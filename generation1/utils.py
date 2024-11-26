import os
import math
import random
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer

cat2id = {c:i for i, c in enumerate([-2, -1, 0, 1])}
id2cat = {i:c for c, i in cat2id.items()}

rating2id = {r:i for i, r in enumerate(range(1, 6))}
id2rating = {i:r for r, i in rating2id.items()}


class MyDataset(Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = pd.read_csv(os.path.join('asap', f'{split}.csv'))
        # if split == 'train':
            # self.data = self.data[::8]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx, 1].replace('\\\\n', '')
        targets = [rating2id[int(self.data.iloc[idx, 2])]]
        for t in self.data.iloc[idx, 3:].values.astype(int).tolist():
            targets.append(cat2id[t])
        return sentence, targets


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        sentences = [d[0] for d in data]
        inputs_tokenized = self.tokenizer(
            sentences,
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        targets_rating = torch.tensor([d[1][0] for d in data]).long()
        targets_cats = torch.tensor([d[1][1:] for d in data]).long()
        return inputs_tokenized, targets_rating, targets_cats.T.reshape(-1), sentences
    

def load_data(batch_size, tokenizer):
    data_collator = DataCollator(tokenizer)

    train_dataset = MyDataset('train')
    val_dataset = MyDataset('dev')
    test_dataset = MyDataset('test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator, num_workers=4)

    return train_loader, val_loader, test_loader


def to_device(batch, device):
    return [item.to(device) for item in batch]


def get_lr_scheduler(optimizer, warmup_steps, total_steps, decrease_mode='cosin'):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            if decrease_mode == 'cosin':
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            elif decrease_mode == 'linear':
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 1.0 - progress
            elif decrease_mode == 'const':
                return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        # 将字典转成JSON格式，确保不转义中文字符，并设置缩进为4
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()