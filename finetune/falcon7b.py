import os
import argparse
import warnings

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, TaskType

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_checkpoint(
    epoch,
    batch_idx,
    model,
    optimizer,
    scaler,
    losses,
    checkpoint_dir='checkpoints_falcon7b',
    local_rank=0,
    max_checkpoints=5
):
    """
    Saves a training checkpoint. Only runs on rank 0.
    """
    if local_rank != 0:
        return

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(
        checkpoint_dir,
        f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pt'
    )

    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'losses': losses
    }, path)
    print(f"Saved checkpoint: {path}")

    # Remove older checkpoints beyond the max limit
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
    ]
    if len(files) > max_checkpoints:
        def parse(f):
            parts = f.rstrip('.pt').split('_')
            try:
                return int(parts[2]), int(parts[4])
            except:
                return 0, 0

        files.sort(key=lambda f: parse(f), reverse=True)
        for old in files[max_checkpoints:]:
            old_path = os.path.join(checkpoint_dir, old)
            os.remove(old_path)
            print(f"Removed old checkpoint: {old_path}")

def find_latest_checkpoint(checkpoint_dir='checkpoints_falcon7b', len_dataloader=None):
    """
    Returns the path of the most recent valid checkpoint, or None.
    """
    if not os.path.exists(checkpoint_dir):
        return None

    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
    ]
    if not files:
        return None

    def parse(f):
        parts = f.rstrip('.pt').split('_')
        try:
            return int(parts[2]), int(parts[4])
        except:
            return 0, 0

    if len_dataloader is not None:
        files = [
            f for f in files
            if parse(f)[1] <= len_dataloader
        ]
        if not files:
            return None

    files.sort(key=lambda f: parse(f), reverse=True)
    return os.path.join(checkpoint_dir, files[0])

def load_checkpoint(checkpoint_path, model, optimizer, scaler, losses, device, local_rank):
    """
    Loads checkpoint into model, optimizer, scaler, and losses. Only on rank 0.
    Returns (epoch, batch_idx).
    """
    if local_rank != 0 or checkpoint_path is None:
        if local_rank == 0 and checkpoint_path is None:
            print("No checkpoint found. Starting from scratch.")
        return 0, 0

    print(f"Loading checkpoint from {checkpoint_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ckpt = torch.load(checkpoint_path, map_location=device)

    model.module.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    losses.extend(ckpt['losses'])

    return ckpt['epoch'], ckpt['batch_idx']

def save_decoded_inputs(dataloader, tokenizer, num_samples=3, rank=0):
    """
    Writes a few decoded input/label pairs to a text file (rank 0 only).
    """
    if rank != 0:
        return

    out_dir = "./trained_falcon7b"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "decoded_input.txt")

    with open(path, "w") as f:
        f.write("--- Decoded Inputs ---\n\n")
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            inp = batch['input_ids'][0]
            lab = batch['labels'][0]
            decoded_inp = tokenizer.decode(inp, skip_special_tokens=True)
            decoded_lab = tokenizer.decode(
                [t if t != -100 else tokenizer.pad_token_id for t in lab],
                skip_special_tokens=True
            )
            f.write(f"Sample {i+1}:\nDecoded Input:\n{decoded_inp}\n")
            f.write(f"Decoded Labels:\n{decoded_lab}\n")
            f.write("\n" + "-"*50 + "\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_rank',
        type=int,
        default=int(os.environ.get('LOCAL_RANK', 0))
    )
    args = parser.parse_args()
    local_rank = args.local_rank

    # Initialize distributed
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Model & tokenizer
    model_name = "tiiuae/falcon-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_8bit=False)
    with record_function("model_loading"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map={"": device}
        )

    # LoRA & DDP
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value", "dense"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # Data
    data_path = '../data/results.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = pd.read_csv(data_path)
    data = data.head(int(len(data) * 0.01))  # first 1%

    class PromptResponseDataset(Dataset):
        def __init__(self, df, tokenizer, max_length=700):
            self.df = df.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            prompt = str(self.df.iloc[idx]['prompt'])
            response = str(self.df.iloc[idx]['response'])
            txt = f"{self.tokenizer.eos_token}{prompt}" \
                  f"{self.tokenizer.eos_token}{response}" \
                  f"{self.tokenizer.eos_token}"
            enc = self.tokenizer(
                txt,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].squeeze()
            attn     = enc['attention_mask'].squeeze()

            eos_id = self.tokenizer.eos_token_id
            eos_positions = (input_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) < 2:
                start = 0
            else:
                start = eos_positions[1].item() + 1

            labels = input_ids.clone()
            labels[:start] = -100

            return {
                'input_ids': input_ids,
                'attention_mask': attn,
                'labels': labels
            }

    dataset = PromptResponseDataset(data, tokenizer)
    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank()
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        pin_memory=True,
        num_workers=4
    )

    if local_rank == 0:
        save_decoded_inputs(dataloader, tokenizer, num_samples=3, rank=0)

    # Checkpoint setup
    latest_ckpt = find_latest_checkpoint(
        checkpoint_dir='checkpoints_falcon7b_7',
        len_dataloader=len(dataloader)
    )

    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5
    )
    scaler = GradScaler()

    losses = []
    start_epoch, start_batch = load_checkpoint(
        latest_ckpt, model, optimizer, scaler, losses, device, local_rank
    )

    # Broadcast resume info
    ep_t = torch.tensor(start_epoch).to(device)
    bt_t = torch.tensor(start_batch).to(device)
    dist.broadcast(ep_t, src=0)
    dist.broadcast(bt_t, src=0)
    start_epoch, start_batch = ep_t.item(), bt_t.item()

    # Training loop
    total_epochs = 100
    grad_acc_steps = 4
    for epoch in range(start_epoch, total_epochs):
        sampler.set_epoch(epoch)
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{total_epochs} started")

        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False) \
            if local_rank == 0 else dataloader

        # Skip if resuming mid-epoch
        if epoch == start_epoch and start_batch > 0:
            for _ in range(start_batch):
                next(iterator, None)

        accumulated_loss = 0.0
        for batch_idx, batch in enumerate(iterator, start=(start_batch if epoch == start_epoch else 0)):
            optimizer.zero_grad(set_to_none=True)
            inputs = batch['input_ids'].to(device, non_blocking=True)
            masks  = batch['attention_mask'].to(device, non_blocking=True)
            lbls   = batch['labels'].to(device, non_blocking=True)

            with record_function("forward_pass"):
                with autocast(device_type='cuda', dtype=torch.float16):
                    out = model(input_ids=inputs, attention_mask=masks, labels=lbls)
                    loss = out.loss / grad_acc_steps

            with record_function("backward_pass"):
                scaler.scale(loss).backward()
                accumulated_loss += loss.item() * grad_acc_steps

            if (batch_idx + 1) % grad_acc_steps == 0 or (batch_idx + 1) == len(dataloader):
                with record_function("optimizer_step"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                if local_rank == 0:
                    current = accumulated_loss / grad_acc_steps
                    losses.append(current)
                    iterator.set_postfix({'loss': f"{current:.4f}"})
                    accumulated_loss = 0.0

                if (batch_idx + 1) % 100 == 0 and local_rank == 0:
                    save_checkpoint(
                        epoch=epoch + 1,
                        batch_idx=batch_idx + 1,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        losses=losses,
                        checkpoint_dir='checkpoints_falcon7b',
                        local_rank=0
                    )

            del loss, out
            torch.cuda.empty_cache()

        start_batch = 0

    # Final save and cleanup
    if local_rank == 0:
        print("Training completed.")
        save_checkpoint(
            epoch=total_epochs,
            batch_idx=len(dataloader),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            losses=losses,
            checkpoint_dir='checkpoints_falcon7b',
            local_rank=0
        )

        out_dir = "./trained_falcon7b"
        os.makedirs(out_dir, exist_ok=True)
        model.module.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        # Save losses and plot
        with open(os.path.join(out_dir, "losses.txt"), "w") as f:
            f.write("\n".join(map(str, losses)))

        plt.figure(figsize=(10, 6))
        plt.plot(losses, label="Training Loss")
        plt.xlabel("Batch Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "loss_curve.png"))
        plt.show()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
