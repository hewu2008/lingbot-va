# encoding:utf8
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import av
import numpy as np
import torch
from tqdm import tqdm

from wan_va.modules.utils import load_vae, load_text_encoder, load_tokenizer


def load_episodes(episodes_file: str) -> list:
    with open(episodes_file, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def discover_camera_keys(dataset_dir: Path) -> list:
    videos_dir = dataset_dir / 'videos' / 'chunk-000'
    if not videos_dir.exists():
        return []
    camera_keys = []
    for item in videos_dir.iterdir():
        if item.is_dir() and (item.name.startswith('observation.images.') or item.name.startswith('observation.image')):
            camera_keys.append(item.name)
    return sorted(camera_keys)


def extract_video_frames(video_path: str, target_fps: int = 10) -> tuple:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    
    # Extract metadata using software decoder
    ori_fps = float(stream.average_rate)
    video_height = stream.height
    video_width = stream.width
    total_frames = stream.frames if stream.frames > 0 else 0
    
    frame_interval = max(1, int(ori_fps / target_fps))
    frames = []
    frame_ids = []
    
    # Loops through video packets and handles software decoding seamlessly
    for idx, frame in enumerate(container.decode(video=0)):
        if idx % frame_interval == 0:
            # PyAV frames can be converted directly to standard RGB NumPy arrays
            frames.append(frame.to_ndarray(format='rgb24'))
            frame_ids.append(idx)
            
    container.close()
    return frames, frame_ids, ori_fps, video_height, video_width


def encode_video_to_latent(
    frames: list,
    vae: torch.nn.Module,
    height: int = 512,
    width: int = 512,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16
) -> dict:
    if not frames:
        return None
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Stack frames to create a tensor with shape: [Frames, Height, Width, Channels]
    video_tensor = torch.from_numpy(np.stack(frames)).float()
    
    # Re-arrange to [Frames, Channels, Height, Width] so F.interpolate handles 2D spatial scaling correctly
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    video_tensor = torch.nn.functional.interpolate(video_tensor, size=(height, width), mode='bilinear', align_corners=False)
    
    # Re-arrange into Wan2.2 VAE's expected 5D shape: [Batch(1), Channels, Frames, Height, Width]
    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
    
    # Normalize pixel inputs to [-1, 1] range
    video_tensor = video_tensor / 255.0 * 2.0 - 1.0
    video_tensor = video_tensor.to(device).to(dtype)
    
    with torch.no_grad():
        enc_out = vae.encode(video_tensor).latent_dist.sample()
        
        # Reshape to [1, 48, 1, 1, 1] to properly broadcast across [B, C, F, H, W]
        latents_mean = torch.tensor(vae.config.latents_mean).to(enc_out.device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std).to(enc_out.device).view(1, -1, 1, 1, 1)
        mu_norm = ((enc_out.float() - latents_mean) * (1.0 / latents_std)).to(dtype)
    
    B, C, F, H, W = mu_norm.shape
    
    # Flatten sequence for downstream transformers
    latent_flat = mu_norm.view(B, C, -1).permute(0, 2, 1).squeeze(0)
    
    return {
        'latent': latent_flat,
        'latent_num_frames': F,
        'latent_height': H,
        'latent_width': W,
    }


def encode_text(
    text: str,
    tokenizer,
    text_encoder,
    max_sequence_length: int = 512,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16
) -> tuple:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    text_inputs = tokenizer(
        text,
        padding='max_length',
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    text_input_ids = text_inputs.input_ids
    mask = text_inputs.attention_mask
    seq_len = mask.gt(0).sum().long()
    
    text_encoder_device = next(text_encoder.parameters()).device
    prompt_embeds = text_encoder(text_input_ids.to(text_encoder_device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
    prompt_embeds = prompt_embeds[:, :seq_len]
    prompt_embeds = torch.cat([
        prompt_embeds,
        prompt_embeds.new_zeros(1, max_sequence_length - seq_len, prompt_embeds.size(2))
    ], dim=1)
    
    return prompt_embeds, text


def process_episode(
    episode: dict,
    dataset_dir: Path,
    vae: torch.nn.Module,
    tokenizer,
    text_encoder,
    camera_keys: list,
    target_fps: int = 10,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    text_encoder_device: torch.device = None,
) -> None:
    episode_index = episode['episode_index']
    action_config = episode.get('action_config', [])
    
    if not action_config:
        start_frame = 0
        end_frame = episode['length']
        action_text = episode['tasks'][0] if episode.get('tasks') else 'unknown task'
        action_config = [{
            'start_frame': start_frame,
            'end_frame': end_frame,
            'action_text': action_text,
        }]
    
    for video_key in camera_keys:
        video_path = dataset_dir / 'videos' / 'chunk-000' / video_key / f'episode_{episode_index:06d}.mp4'
        
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue
        
        frames, frame_ids, ori_fps, video_height, video_width = extract_video_frames(str(video_path), target_fps)
        
        if not frames:
            print(f"No frames extracted from: {video_path}")
            continue
        
        for action_segment in action_config:
            start_frame = action_segment['start_frame']
            end_frame = action_segment['end_frame']
            action_text = action_segment['action_text']
            
            frame_mask = [(fid >= start_frame) and (fid < end_frame) for fid in frame_ids]
            segment_frame_ids = [fid for fid, mask in zip(frame_ids, frame_mask) if mask]
            segment_frames = [frame for frame, mask in zip(frames, frame_mask) if mask]
            
            if not segment_frames:
                print(f"No frames in segment [{start_frame}, {end_frame}) for episode {episode_index}")
                continue
            
            latent_info = encode_video_to_latent(
                segment_frames, vae,
                height=480, width=640,
                device=device, dtype=dtype
            )
            
            text_emb, _ = encode_text(action_text, tokenizer, text_encoder, device=text_encoder_device or device, dtype=dtype)
            
            output_dict = {
                'latent': latent_info['latent'],
                'latent_num_frames': latent_info['latent_num_frames'],
                'latent_height': latent_info['latent_height'],
                'latent_width': latent_info['latent_width'],
                'video_num_frames': len(segment_frames),
                'video_height': video_height,
                'video_width': video_width,
                'text_emb': text_emb,
                'text': action_text,
                'frame_ids': segment_frame_ids,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'fps': target_fps,
                'ori_fps': ori_fps,
            }
            
            latent_dir = dataset_dir / 'latents' / 'chunk-000' / video_key
            latent_dir.mkdir(parents=True, exist_ok=True)
            
            latent_filename = f'episode_{episode_index:06d}_{start_frame}_{end_frame}.pth'
            latent_path = latent_dir / latent_filename
            
            torch.save(output_dict, str(latent_path))
            print(f"Saved latent: {latent_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract Wan2.2 VAE latents from LeRobot dataset videos')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to LeRobot dataset directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Wan2.2 pretrained model')
    parser.add_argument('--target_fps', type=int, default=10, help='Target FPS for latent extraction')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'], help='Data type')
    parser.add_argument('--enable_offload', action='store_true', help='Offload text_encoder to CPU to save VRAM')
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_encoder_device = torch.device('cpu') if args.enable_offload else device
    
    print(f"Loading Wan2.2 VAE model from: {args.model_path}")
    
    vae = load_vae(
        os.path.join(args.model_path, 'vae'),
        torch_dtype=dtype,
        torch_device=device,
    )
    
    tokenizer = load_tokenizer(os.path.join(args.model_path, 'tokenizer'))
    text_encoder = load_text_encoder(
        os.path.join(args.model_path, 'text_encoder'),
        torch_dtype=dtype,
        torch_device=text_encoder_device,
    )
    
    camera_keys = discover_camera_keys(dataset_dir)
    if not camera_keys:
        camera_keys = ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']
    print(f"Discovered camera keys: {camera_keys}")
    
    episodes_file = dataset_dir / 'meta' / 'episodes_lingbot.jsonl'
    episodes = load_episodes(str(episodes_file))
    print(f"Loaded {len(episodes)} episodes")
    
    for episode in tqdm(episodes, desc='Processing episodes'):
        process_episode(
            episode, dataset_dir,
            vae, tokenizer, text_encoder,
            camera_keys,
            target_fps=args.target_fps,
            device=device, dtype=dtype,
            text_encoder_device=text_encoder_device,
        )
        break
    
    print("Latent extraction completed successfully!")


if __name__ == '__main__':
    main()