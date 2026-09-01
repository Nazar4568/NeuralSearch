import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.ml.models.retriever import TextRetriever
from src.ml.models.weight_transfer import load_huggingface_weights
from src.ml.training.train_retriever import train
from src.ml.training.dataset import RetrievalDataset, CollateFunction
from src.ml.training.losses import InfoNCELoss
from src.data.tokenizer import TextTokenizer


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    custom_model = TextRetriever(
        vocab_size=30522, d_model=384, num_heads=12, d_ff=1536, num_layers=2, max_seq_len=512
    )
    model = load_huggingface_weights(custom_model, "sentence-transformers/all-MiniLM-L6-v2")

    active_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(active_parameters, lr=2e-5, weight_decay=0.01)

    loss_fn = InfoNCELoss(temperature=0.05)
    tokenizer = TextTokenizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
    collate_fn = CollateFunction(tokenizer=tokenizer)

    dataset_path = "data/processed/code_triplets.jsonl"

    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Run build_train_dataset.py first.")
        return

    train_dataset = RetrievalDataset(dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=1,
        save_path="models/retriever_weights.pth"
    )


if __name__ == "__main__":
    main()