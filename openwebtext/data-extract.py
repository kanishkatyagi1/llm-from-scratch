from datasets import load_dataset
from tqdm import tqdm
import random

print("USING STREAMING MODE")

dataset = load_dataset(
    "Skylion007/openwebtext",
    split="train",
    streaming=True,
    trust_remote_code=True  
)

texts = []
print("Collecting first 10000 samples...")

for i, sample in enumerate(tqdm(dataset, total=10000)):  # ← adds a progress bar while collecting
    texts.append(sample["text"])
    if i >= 9999:
        break

print(f"Collected {len(texts)} documents")

random.shuffle(texts)

split_idx = int(0.9 * len(texts))
train_texts = texts[:split_idx]
val_texts = texts[split_idx:]

vocab = set()

print("Writing training file...")
with open("output_train.txt", "w", encoding="utf-8") as train_file:
    for text in tqdm(train_texts):
        train_file.write(text + "\n")
        vocab.update(text)

print("Writing validation file...")
with open("output_val.txt", "w", encoding="utf-8") as val_file:
    for text in tqdm(val_texts):
        val_file.write(text + "\n")
        vocab.update(text)

print("Writing vocabulary file...")
with open("vocab.txt", "w", encoding="utf-8") as vocab_file:
    for ch in sorted(vocab):
        vocab_file.write(ch + "\n")

print("Done!")
print(f"Train docs: {len(train_texts)}, Val docs: {len(val_texts)}")
print(f"Vocabulary size: {len(vocab)}")