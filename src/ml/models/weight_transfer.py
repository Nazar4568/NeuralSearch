import torch
from transformers import AutoModel
from src.ml.models.retriever import TextRetriever


def load_huggingface_weights(
        custom_model: TextRetriever,
        hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> TextRetriever:
    print(f"Downloading pre-trained weights from {hf_model_name}...")
    hf_model = AutoModel.from_pretrained(hf_model_name)

    hf_state_dict = hf_model.state_dict()
    custom_state_dict = custom_model.state_dict()

    print("Mapping weights to custom architecture...")

    custom_state_dict["embedder.word_embeddings.weight"] = hf_state_dict["embeddings.word_embeddings.weight"]
    custom_state_dict["embedder.position_embeddings.weight"] = hf_state_dict["embeddings.position_embeddings.weight"]
    custom_state_dict["embedder.LayerNorm.weight"] = hf_state_dict["embeddings.LayerNorm.weight"]
    custom_state_dict["embedder.LayerNorm.bias"] = hf_state_dict["embeddings.LayerNorm.bias"]

    for i in range(2):
        prefix_hf = f"encoder.layer.{i}."
        prefix_custom = f"module_list.{i}."

        custom_state_dict[f"{prefix_custom}attention.q_proj.weight"] = hf_state_dict[
            f"{prefix_hf}attention.self.query.weight"]
        custom_state_dict[f"{prefix_custom}attention.q_proj.bias"] = hf_state_dict[
            f"{prefix_hf}attention.self.query.bias"]
        custom_state_dict[f"{prefix_custom}attention.k_proj.weight"] = hf_state_dict[
            f"{prefix_hf}attention.self.key.weight"]
        custom_state_dict[f"{prefix_custom}attention.k_proj.bias"] = hf_state_dict[
            f"{prefix_hf}attention.self.key.bias"]
        custom_state_dict[f"{prefix_custom}attention.v_proj.weight"] = hf_state_dict[
            f"{prefix_hf}attention.self.value.weight"]
        custom_state_dict[f"{prefix_custom}attention.v_proj.bias"] = hf_state_dict[
            f"{prefix_hf}attention.self.value.bias"]

        custom_state_dict[f"{prefix_custom}attention.out_proj.weight"] = hf_state_dict[
            f"{prefix_hf}attention.output.dense.weight"]
        custom_state_dict[f"{prefix_custom}attention.out_proj.bias"] = hf_state_dict[
            f"{prefix_hf}attention.output.dense.bias"]

        custom_state_dict[f"{prefix_custom}FeedForward.linear1.weight"] = hf_state_dict[f"{prefix_hf}intermediate.dense.weight"]
        custom_state_dict[f"{prefix_custom}FeedForward.linear1.bias"] = hf_state_dict[f"{prefix_hf}intermediate.dense.bias"]
        custom_state_dict[f"{prefix_custom}FeedForward.linear2.weight"] = hf_state_dict[f"{prefix_hf}output.dense.weight"]
        custom_state_dict[f"{prefix_custom}FeedForward.linear2.bias"] = hf_state_dict[f"{prefix_hf}output.dense.bias"]

        custom_state_dict[f"{prefix_custom}norm1.weight"] = hf_state_dict[
            f"{prefix_hf}attention.output.LayerNorm.weight"]
        custom_state_dict[f"{prefix_custom}norm1.bias"] = hf_state_dict[f"{prefix_hf}attention.output.LayerNorm.bias"]
        custom_state_dict[f"{prefix_custom}norm2.weight"] = hf_state_dict[f"{prefix_hf}output.LayerNorm.weight"]
        custom_state_dict[f"{prefix_custom}norm2.bias"] = hf_state_dict[f"{prefix_hf}output.LayerNorm.bias"]

    custom_model.load_state_dict(custom_state_dict)

    for name, param in custom_model.named_parameters():
        if "embeddings" in name:
            param.requires_grad = False

    print("Successfully injected HuggingFace weights and froze embedding layers.")
    return custom_model