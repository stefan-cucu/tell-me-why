
import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
import argparse
from tqdm import tqdm

from common import MLPClassifier, ReportEmbedder, load_json_report


class MalwareDataset(Dataset):
    
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = torch.load(self.embeddings[idx], map_location='cpu')
        label = self.labels[idx]
        return emb, torch.LongTensor([label])[0]


def evaluate(model, dataloader, criterion, device, phase="Evaluating"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / max(len(dataloader), 1), accuracy, f1, all_preds, all_labels


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch}] Training")
    for embeddings, labels in pbar:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.2f}%"
        )

    return total_loss / len(dataloader), 100 * correct / total


def load_split_embeddings(split_file, split_name, embedder, cache_dir):
    with open(split_file, 'r') as f:
        splits = json.load(f)

    paths = splits.get(split_name, [])
    emb_paths = []
    labels = []
    to_embed = []  

    os.makedirs(cache_dir, exist_ok=True)

    for path in paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        label = 0 if "goodware_dataset" in path else 1
        cache_path = os.path.join(cache_dir, f"{basename}_{label}.pt")

        if os.path.exists(cache_path):
            emb_paths.append(cache_path)
            labels.append(label)
        else:
            to_embed.append((path, label, cache_path))

    if to_embed:
        skipped = 0
        print(f"    Embedding {len(to_embed)} uncached {split_name} samples...")
        for path, label, cache_path in tqdm(to_embed, desc=f"Embedding {split_name}"):
            try:
                content = load_json_report(path)
                if not content:
                    continue
                
                
                if len(content) / 3 >= 30000:
                    skipped += 1
                    continue
                embedding = embedder.embed(content)
                torch.save(embedding, cache_path)
                emb_paths.append(cache_path)
                labels.append(label)
            except Exception as e:
                print(f"    Skipping {path}: {e}")
        if skipped:
            print(f"    Skipped {skipped} reports >= 32k tokens")

    return emb_paths, labels


def load_adversarial_embeddings(adversarial_dirs, embedder, cache_dir):
    
    os.makedirs(cache_dir, exist_ok=True)

    cached_paths = []
    labels = []

    for d in adversarial_dirs:
        files = sorted(glob.glob(os.path.join(d, "evaded_*.json")))
        print(f"Found {len(files)} evaded logs in {d}")

        for fp in tqdm(files, desc=f"Embedding {os.path.basename(d)}"):
            file_id = os.path.splitext(os.path.basename(fp))[0]
            emb_cache_path = os.path.join(cache_dir, f"{file_id}_1.pt")  

            if os.path.exists(emb_cache_path):
                cached_paths.append(emb_cache_path)
                labels.append(1)
                continue

            try:
                with open(fp, 'r') as f:
                    data = json.load(f)

                
                if 'evaded_data' in data and data['evaded_data'] is not None:
                    report = data['evaded_data']
                elif 'summary' in data:
                    report = data['summary']
                else:
                    report = data

                text = json.dumps(report) if isinstance(report, dict) else str(report)

                
                if len(text) / 4 >= 32000:
                    continue

                embedding = embedder.embed(text)
                torch.save(embedding, emb_cache_path)
                cached_paths.append(emb_cache_path)
                labels.append(1)
            except Exception as e:
                print(f"  Skipping {fp}: {e}")

    return cached_paths, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adversarial-dirs", nargs="+", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    embedder = ReportEmbedder(args.embedding_model, args.embedding_device)

    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    
    print(f"\nLoading original train split from {args.splits}...")
    train_emb_paths, train_labels = load_split_embeddings(
        args.splits, "train", embedder, args.cache
    )
    print(f"  Original train: {len(train_emb_paths)} samples "
          f"({sum(1 for l in train_labels if l==0)} goodware, {sum(1 for l in train_labels if l==1)} malware)")

    val_emb_paths, val_labels = load_split_embeddings(
        args.splits, "val", embedder, args.cache
    )
    print(f"  Original val:   {len(val_emb_paths)} samples "
          f"({sum(1 for l in val_labels if l==0)} goodware, {sum(1 for l in val_labels if l==1)} malware)")

    test_emb_paths, test_labels = load_split_embeddings(
        args.splits, "test", embedder, args.cache
    )
    print(f"  Original test:  {len(test_emb_paths)} samples")

    if not train_emb_paths:
        print("ERROR: No training embeddings were generated.")
        return

    
    print(f"\nEmbedding evaded logs from: {args.adversarial_dirs}")
    evaded_emb_paths, evaded_labels = load_adversarial_embeddings(
        args.adversarial_dirs,
        embedder,
        args.cache,
    )
    print(f"  Evaded samples: {len(evaded_emb_paths)} (all labeled malware)")

    
    combined_train_paths = train_emb_paths + evaded_emb_paths
    combined_train_labels = train_labels + evaded_labels
    print(f"\n  Combined train: {len(combined_train_paths)} samples "
          f"({sum(1 for l in combined_train_labels if l==0)} goodware, "
          f"{sum(1 for l in combined_train_labels if l==1)} malware)")

    
    train_dataset = MalwareDataset(combined_train_paths, combined_train_labels)
    val_dataset = MalwareDataset(val_emb_paths, val_labels)
    test_dataset = MalwareDataset(test_emb_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    
    print(f"\nLoading classifier from {args.classifier}...")
    model = MLPClassifier(1024).to(device)
    model.load_state_dict(torch.load(args.classifier, map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    
    print("\n--- Baseline (before fine-tuning) ---")
    _, base_val_acc, base_val_f1, _, _ = evaluate(model, val_loader, criterion, device, phase="Baseline Val")
    print(f"  Val Acc: {base_val_acc:.2%} | Val F1: {base_val_f1:.4f}")

    
    if evaded_emb_paths:
        evaded_dataset = MalwareDataset(evaded_emb_paths, evaded_labels)
        evaded_loader = DataLoader(evaded_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        _, evaded_acc, _, evaded_preds, evaded_true = evaluate(model, evaded_loader, criterion, device, phase="Baseline Evaded")
        evaded_correct = sum(1 for p in evaded_preds if p == 1)
        print(f"  Evaded detection (before): {evaded_correct}/{len(evaded_preds)} "
              f"({evaded_correct/max(len(evaded_preds),1):.1%} correctly detected as malware)")

    
    best_val_f1 = 0.0

    print(f"\nFine-tuning for {args.epochs} epochs on {len(combined_train_paths)} samples...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved new best model (Val F1: {val_f1:.4f})")

    
    if best_val_f1 == 0.0:
        torch.save(model.state_dict(), args.output)
        print(f"  -> Saved final model (no validation improvement)")

    
    print("\n--- Final Test Evaluation ---")
    model.load_state_dict(torch.load(args.output, map_location=device))

    test_loss, test_acc, test_f1, preds, final_labels = evaluate(model, test_loader, criterion, device, phase="Final Testing")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Test F1 Macro: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(final_labels, preds, target_names=["goodware", "malware"]))

    
    if evaded_emb_paths:
        _, _, _, evaded_preds_after, _ = evaluate(model, evaded_loader, criterion, device, phase="Post-train Evaded")
        evaded_correct_after = sum(1 for p in evaded_preds_after if p == 1)
        print(f"\nEvaded detection (after): {evaded_correct_after}/{len(evaded_preds_after)} "
              f"({evaded_correct_after/max(len(evaded_preds_after),1):.1%} correctly detected as malware)")
        print(f"Improvement: {evaded_correct_after - evaded_correct:+d} samples")

    print(f"\nBaseline Val F1: {base_val_f1:.4f} -> Final Val F1: {best_val_f1:.4f} (delta: {best_val_f1 - base_val_f1:+.4f})")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
