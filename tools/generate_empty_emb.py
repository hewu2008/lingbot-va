import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wan_va'))
from modules.utils import load_text_encoder, load_tokenizer


def generate_empty_emb(model_path: str, output_path: str):
    """
    Generate empty_emb.pt file by encoding an empty string.
    
    Args:
        model_path: Path to Wan2.2 pretrained model
        output_path: Path to save empty_emb.pt
    """
    print(f"Loading tokenizer and text encoder from: {model_path}")
    
    tokenizer = load_tokenizer(os.path.join(model_path, 'tokenizer'))
    text_encoder = load_text_encoder(
        os.path.join(model_path, 'text_encoder'),
        torch_dtype=torch.bfloat16,
        torch_device=torch.device('cpu'),
    )
    
    max_sequence_length = 512
    
    text_inputs = tokenizer(
        "",
        padding='max_length',
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    text_input_ids = text_inputs.input_ids
    
    with torch.no_grad():
        text_encoder_device = next(text_encoder.parameters()).device
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder_device)).last_hidden_state
    
    prompt_embeds = prompt_embeds.squeeze(0).to(torch.bfloat16)
    
    print(f"Generated empty embedding shape: {prompt_embeds.shape}")
    print(f"Data type: {prompt_embeds.dtype}")
    print(f"Mean: {prompt_embeds.mean().item():.6f}")
    print(f"Std: {prompt_embeds.std().item():.6f}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(prompt_embeds, str(output_path))
    print(f"Saved empty embedding to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate empty_emb.pt for CFG training')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Wan2.2 pretrained model')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for empty_emb.pt')
    args = parser.parse_args()
    
    generate_empty_emb(args.model_path, args.output_path)