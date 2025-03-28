import os
import torch
import sentencepiece as spm
import gcsfs
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2LMHeadModel, AdamW, get_scheduler
from tqdm import tqdm

# Define model path - Path to your existing model
MODEL_DIR = "/home/apthy/Dokumente/TSModel2/TS2gpt2_finetuned_dm_tokenizer"

# Define the new directory for all outputs
NEW_BASE_DIR = "/home/apthy/Dokumente/TSModel2_continue"

# Define where to save new model and checkpoints
OUTPUT_DIR = os.path.join(NEW_BASE_DIR, "TS2gpt2_finetuned_dm_tokenizer_4epochs")
CHECKPOINT_DIR = os.path.join(NEW_BASE_DIR, "checkpoints")

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the new directory if it doesn't exist
os.makedirs(NEW_BASE_DIR, exist_ok=True)

# Set up logging file for the continued training
log_filename = os.path.join(NEW_BASE_DIR, "TS2training_continued_log.txt")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)
logging.info("Starting continued training session")
logging.info(f"Model input directory: {MODEL_DIR}")
logging.info(f"Output directory: {OUTPUT_DIR}")
logging.info(f"Checkpoint directory: {CHECKPOINT_DIR}")

# Load the DeepMind Tokenizer
fs = gcsfs.GCSFileSystem()
TOKENIZER_PATH = 'gs://transformer-ngrams/32768.model'
with fs.open(TOKENIZER_PATH, 'rb') as f:
    tokenizer_sp = spm.SentencePieceProcessor(model_proto=f.read())

# Load the previously saved model
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.to(device)
logging.info(f"Loaded model from {MODEL_DIR}")

# Load TinyStories again
TINYSTORIES_TRAINING_DATA_PATH = 'gs://transformer-ngrams/TinyStories/training_data/'
tiny_files = [f'gs://{file}' for file in fs.ls(TINYSTORIES_TRAINING_DATA_PATH)]
logging.info(f"Found {len(tiny_files)} TinyStories files")

tiny_dfs = [pd.read_parquet(fs.open(file, 'rb')) for file in tiny_files]
df_tiny = pd.concat(tiny_dfs)
tokens_list = df_tiny["tokens"].tolist()

# Tokenization & Chunking (1024 Tokens)
tokenized_chunks = []
max_length = 1024

for tokens in tokens_list:
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i+max_length]
        if len(chunk) == max_length:
            tokenized_chunks.append(torch.tensor(chunk))

logging.info(f"Created {len(tokenized_chunks)} chunks for training")

# Dataset & Split
class GPT2Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        return input_ids, input_ids.clone()

dataset = GPT2Dataset(tokenized_chunks)
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch]).long()
    labels = torch.stack([item[1] for item in batch]).long()
    return input_ids, labels

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

# Optimizer setup
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
num_epochs = 1  # Just one more epoch
steps_per_epoch = len(train_loader)
num_training_steps = steps_per_epoch * num_epochs
checkpoint_frequency = steps_per_epoch // 5  # Save checkpoint 5 times per epoch

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

logging.info(f"Starting continued training for {num_epochs} epoch")
logging.info(f"Total steps: {num_training_steps}, checkpoint frequency: {checkpoint_frequency}")

# Training + Validation
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0

    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch 4 [Train]")  # Label as epoch 4 since it follows 3 previous epochs
    for step, batch in train_loop:
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())
        
        # Saving checkpoints
        if (step + 1) % checkpoint_frequency == 0:
            # Use epoch 4 since we're continuing from 3 previous epochs
            checkpoint_filename = f"epoch_4_step_{step+1}.pth"
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
            torch.save({
                'epoch': 4,  # This is the 4th epoch overall
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / (step + 1),
            }, checkpoint_path)

            logging.info(f"Checkpoint saved at {checkpoint_path}, Train Loss: {train_loss / (step + 1):.4f}")

    avg_train_loss = train_loss / len(train_loader)
    logging.info(f"Epoch 4 completed | Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch 4 [Val]")
        for batch in val_loop:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    logging.info(f"Epoch 4 completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# Save the updated model
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)

# Copy the tokenizer to the new save location
with fs.open(TOKENIZER_PATH, 'rb') as f:
    with open(os.path.join(OUTPUT_DIR, "tokenizer.model"), "wb") as out_f:
        out_f.write(f.read())

logging.info(f"Updated model saved in: {OUTPUT_DIR}")
print(f"Updated model saved in: {OUTPUT_DIR}")
