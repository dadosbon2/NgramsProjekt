#tinystories Model mit 124M weil training viel zu lange dauert
import os
import torch
import sentencepiece as spm
import gcsfs
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2LMHeadModel, AdamW, get_scheduler
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#0 Set up logging file to check later
log_filename = "TS2training_log_.txt"
if os.path.exists(log_filename):
    os.remove(log_filename)  # Remove old log file to start fresh

logging.basicConfig(
    filename=log_filename,  # Save logs to this file
    level=logging.INFO,  # Only log INFO and above (ignore DEBUG logs)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    filemode="w"  # Overwrite the file each run (use "a" to append instead)
)

# 1. DeepMind Tokenizer laden
fs = gcsfs.GCSFileSystem()
TOKENIZER_PATH = 'gs://transformer-ngrams/32768.model'
with fs.open(TOKENIZER_PATH, 'rb') as f:
    tokenizer_sp = spm.SentencePieceProcessor(model_proto=f.read())

# Test Tokenisierung
print("Tokenizer Test:", tokenizer_sp.encode("Once upon a time"))

# 2. GPT-2 vortrainiert laden + Embeddings anpassen
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(32768)  # Anpassung an Tokenizer
model.to(device)

# 3. TinyStories laden
TINYSTORIES_TRAINING_DATA_PATH = 'gs://transformer-ngrams/TinyStories/training_data/'
tiny_files = [f'gs://{file}' for file in fs.ls(TINYSTORIES_TRAINING_DATA_PATH)]
tiny_dfs = [pd.read_parquet(fs.open(file, 'rb')) for file in tiny_files]
df_tiny = pd.concat(tiny_dfs)
# texts = df_tiny["text"].tolist()    removed due not being necessary
tokens_list = df_tiny["tokens"].tolist()

# 4. Tokenisierung & Chunking (1024 Tokens)
tokenized_chunks = []
max_length = 1024
 
 #remove enconding to the texts again
for tokens in tokens_list:
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i+max_length]
        if len(chunk) == max_length:
            tokenized_chunks.append(torch.tensor(chunk))

# 5. Dataset & Split
class GPT2Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        return input_ids, input_ids.clone()

dataset = GPT2Dataset(tokenized_chunks)
print("Dataset sample:", dataset[0])
print("Dataset sample type:", type(dataset[0]))


train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def collate_fn(batch):
    print("Collate function received:", batch)  # Debugging print
    input_ids = torch.stack([item[0] for item in batch]).long()
    labels = torch.stack([item[1] for item in batch]).long()
    return input_ids, labels

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

# 6. Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
num_epochs = 3
steps_per_epoch = len(train_loader)
num_training_steps = len(train_loader) * num_epochs
checkpoint_frequency = steps_per_epoch // 5  # Save checkpoint 5 times per epoch
num_warmup_steps=0 #Since we are fine tuning
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


# 7. Training + Validation Loss
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0

    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} [Train]")
    for step, batch in train_loop:

        input_ids, labels = batch[0], batch[1]
        input_ids, labels = input_ids.to(device), labels.to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())
        
        #saving checkpoints for possible context experiments
        if (step + 1) % checkpoint_frequency == 0:
            checkpoint_filename = f"epoch_{epoch+1}_step_{step+1}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = os.path.join("checkpoints", checkpoint_filename)
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / (step + 1),
            }, checkpoint_path)

            # Log the checkpoint info (loss at that point)
            logging.info(f"Checkpoint saved at {checkpoint_filename}, Train Loss: {train_loss / (step + 1):.4f}")

    avg_train_loss = train_loss / len(train_loader)
    logging.info(f"Epoch {epoch+1} abgeschlossen | Train Loss: {avg_train_loss:.4f}")


    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        for batch in val_loop:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    logging.info(f"Epoch {epoch+1} abgeschlossen | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}") #Here we dont use the print funcitonn anymore as we dont know how to create a remote session like tmux or screen for the gpu cluster. All the reports will be save a txt file

# 8. Speichern
save_path = os.path.join(os.getcwd(), "TS2gpt2_finetuned_dm_tokenizer")
model.save_pretrained(save_path)

# Tokenizer speichern (Raw File)
with fs.open(TOKENIZER_PATH, 'rb') as f:
    with open(os.path.join(save_path, "tokenizer.model"), "wb") as out_f:
        out_f.write(f.read())

print(f"Modell gespeichert in: {save_path}")
