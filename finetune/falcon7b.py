#!/usr/bin/env python3
import os

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, TaskType

# Hardcoded configuration (no CLI args needed)
CSV_PATH = '../data/results.csv'
CHECKPOINT_DIR = 'checkpoints_falcon7b'
BATCH_SIZE = 2
EPOCHS = 3
MAX_LENGTH = 700
MODEL_NAME = 'tiiuae/falcon-7b'


def save_checkpoint(epoch, batch_idx, model, optimizer, scaler, checkpoint_dir, max_checkpoints=5):
    """
    Save a training checkpoint (only on rank 0).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch}_batch_{batch_idx}.pt")
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state': model.module.state_dict(),
        'optim_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict()
    }, path)
    print(f"[Rank 0] Saved checkpoint: {path}")

    # cleanup old
    files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('ckpt_')],
                   key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    if len(files) > max_checkpoints:
        for old in files[:-max_checkpoints]:
            os.remove(os.path.join(checkpoint_dir, old))


def find_latest_checkpoint(checkpoint_dir, max_batch_idx=None):
    """
    Return latest checkpoint path, or None.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    cks = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not cks:
        return None
    def keyfn(fn):
        parts = fn.replace('.pt','').split('_')
        try:
            return (int(parts[2]), int(parts[4]))
        except:
            return (0,0)
    cks = sorted(cks, key=keyfn, reverse=True)
    for ck in cks:
        if max_batch_idx is None:
            return os.path.join(checkpoint_dir, ck)
        _, batch = keyfn(ck)
        if batch <= max_batch_idx:
            return os.path.join(checkpoint_dir, ck)
    return None


def load_checkpoint(path, model, optimizer, scaler, device, local_rank):
    """
    Load checkpoint into model/optimizer/scaler. Returns (epoch, batch_idx).
    """
    if not path or local_rank != 0:
        return 0, 0
    print(f"[Rank 0] Loading checkpoint from {path}")
    chk = torch.load(path, map_location=device)
    model.module.load_state_dict(chk['model_state'])
    optimizer.load_state_dict(chk['optim_state'])
    scaler.load_state_dict(chk['scaler_state'])
    return chk['epoch'], chk['batch_idx']


class PromptResponseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = str(self.df.loc[idx, 'prompt'])
        response = str(self.df.loc[idx, 'response'])
        text = f"{self.tokenizer.eos_token}{prompt}{self.tokenizer.eos_token}{response}{self.tokenizer.eos_token}"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)

        # mask prompt tokens
        eos_id = self.tokenizer.eos_token_id
        eos_positions = (input_ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) >= 2:
            start = eos_positions[1].item() + 1
        else:
            start = 0
        labels = input_ids.clone()
        labels[:start] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def main():
    # DDP setup
   # local_rank = int(os.environ.get('LOCAL_RANK', 0))
    local_rank = 0
    #dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    device = torch.device(f'cuda:{local_rank}')
   

    # Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={'': device}
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=['query_key_value', 'dense'],
        lora_dropout=0.1, bias='none',
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    # TODO actiavte thjis when i am using more GPUs
    #model = DistributedDataParallel(model, device_ids=[local_rank])
    model.to(device)

    # Data
    df = pd.read_csv(CSV_PATH)
    dataset = PromptResponseDataset(df, tokenizer)
    # TODO
    #sampler = DistributedSampler(dataset, shuffle=True)
    # dataloader = DataLoader(dataset,
    #                          batch_size=BATCH_SIZE,
    #                          sampler=sampler,
    #                          pin_memory=True,
    #                          num_workers=4)
    dataloader = DataLoader(dataset,
                             batch_size=BATCH_SIZE,
                             pin_memory=True,
                             num_workers=4)

    # Optimizer & scaler
    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5
    )
    scaler = GradScaler()

    # Resume
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    latest = find_latest_checkpoint(CHECKPOINT_DIR, len(dataloader))
    start_epoch, start_batch = load_checkpoint(latest, model, optimizer, scaler, device, local_rank)

    # Train
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader,
                                         start=start_batch if epoch == start_epoch else 0):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            with autocast():
                loss = model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels).loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if local_rank == 0 and (batch_idx + 1) % 100 == 0:
                save_checkpoint(epoch, batch_idx+1, model, optimizer, scaler, CHECKPOINT_DIR)

        start_batch = 0

    # Final save & cleanup
    if local_rank == 0:
        save_checkpoint(EPOCHS, len(dataloader), model, optimizer, scaler, CHECKPOINT_DIR)
        out = 'trained_model'
        os.makedirs(out, exist_ok=True)
        model.module.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print(f"Model and tokenizer saved to '{out}'")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()

