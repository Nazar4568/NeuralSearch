import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str,
        epochs: int = 3,
        save_path: str = "models/retriever_weights.pth"
) -> None:
    """
    Executes the training and validation loops for the retrieval model.
    Saves the best model weights based on validation loss.
    """
    model = model.to(device)
    best_val_loss = float('inf')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for batch in progress_bar:
            query_ids = batch["query"]["input_ids"].to(device)
            query_mask = batch["query"]["attention_mask"].to(device)

            pos_ids = batch["positive"]["input_ids"].to(device)
            pos_mask = batch["positive"]["attention_mask"].to(device)

            query_embeddings = model(query_ids, query_mask)
            pos_embeddings = model(pos_ids, pos_mask)

            loss = loss_fn(query_embeddings, pos_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                q_ids = val_batch["query"]["input_ids"].to(device)
                q_mask = val_batch["query"]["attention_mask"].to(device)

                p_ids = val_batch["positive"]["input_ids"].to(device)
                p_mask = val_batch["positive"]["attention_mask"].to(device)

                q_emb = model(q_ids, q_mask)
                p_emb = model(p_ids, p_mask)

                val_loss = loss_fn(q_emb, p_emb)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)