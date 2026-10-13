import torch
import os
import json
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.dataset import MILVideoDataset, MILVideoDatasetRope
from dino.utils import make_val_transform_dino
from dino.train_dino_transformer import DinoSelfAttention
from dino.train_dino_transformer_rope import DinoSelfAttention as DinoSelfAttentionRope


def unnormalize_img(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Unnormalizes a tensor image for visualization."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)


def plot_video_attention(inputs, weights, video_name, out_dir):
    """
    Plots a bar chart of the 32 attention weights and a grid of the corresponding 32 frames.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(20, 13))

    # Panel 1: Bar chart of attention weights
    ax_bar = plt.subplot2grid((5, 8), (0, 0), colspan=8)
    frames_idx = np.arange(len(weights))
    ax_bar.bar(frames_idx, weights, color='royalblue', edgecolor='black')
    ax_bar.axhline(1.0 / len(weights), color='red', linestyle='--', label="Uniform (1/32)")
    ax_bar.set_title(f"Attention Weights for {video_name}", fontsize=14, fontweight='bold')
    ax_bar.set_xlabel("Frame Index")
    ax_bar.set_ylabel("Attention Weight")
    ax_bar.set_xlim(-0.5, 31.5)
    ax_bar.legend()

    # Panel 2: 32 Frames Grid
    for i in range(32):
        ax = plt.subplot2grid((5, 8), (1 + i // 8, i % 8))
        img = unnormalize_img(inputs[i]).permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(f"Fr {i} | W: {weights[i]:.3f}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{video_name}_attention.png"), dpi=150)
    plt.close(fig)


def analyze_attention_distribution(attention_records):
    """Calculates Entropy and Max Weight to summarize attention behavior."""
    print("\n--- Transformer Attention Overview ---")
    if not attention_records:
        return

    all_max_weights = []
    all_entropies = []

    for rec in attention_records:
        w = np.array(rec["weights"])
        all_max_weights.append(np.max(w))
        # Entropy: -sum(p * log(p)). Lower entropy = spikier attention.
        entropy = -np.sum(w * np.log(w + 1e-12))
        all_entropies.append(entropy)

    avg_max = np.mean(all_max_weights)
    max_max = np.max(all_max_weights)
    avg_entropy = np.mean(all_entropies)

    # Uniform attention over 32 frames has max ~0.031 and entropy ~3.46
    print(f"Mean Max Attention Weight per video: {avg_max:.4f} (Uniform would be ~0.0312)")
    print(f"Highest Single-Frame Attention found: {max_max:.4f}")
    print(f"Mean Attention Entropy: {avg_entropy:.4f} (Max possible ~3.465)")

    if avg_max > 0.15:
        print("Verdict: The model is highly selective/spiky, focusing on specific frames.")
    elif avg_max < 0.06:
        print("Verdict: The model relies on a relatively uniform global pooling context.")
    else:
        print("Verdict: The model shows moderate selectivity with some focal points.")
    print("--------------------------------------\n")


def run_inference_transformer(model, loader, device, run_name, base_out_dir):
    model.eval()
    attention_records = []
    dataset_samples = loader.dataset.data
    attn_dir = os.path.join(base_out_dir, f"attention_vis_{run_name}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Transformer Inference (Binary)")):
            inputs = batch[0].squeeze(0).to(device)
            video_name = dataset_samples[i]['vid_name']

            # Capture weights returned by forward()
            _, weights = model(inputs)

            # Move weights to CPU for plotting & logging
            w_np = weights.squeeze(0).cpu().numpy()

            attention_records.append({"video": video_name, "weights": w_np})

            # Generate visualization
            plot_video_attention(inputs.cpu(), w_np, video_name, attn_dir)

    analyze_attention_distribution(attention_records)


def run_inference_transformer_all(model, loader, device, run_name, base_out_dir):
    model.eval()
    attention_records = []
    dataset_samples = loader.dataset.data
    attn_dir = os.path.join(base_out_dir, f"attention_vis_{run_name}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Transformer Inference (Multi-class)")):
            inputs = batch[0].squeeze(0).to(device)
            video_name = dataset_samples[i]['vid_name']

            _, weights = model(inputs)
            w_np = weights.squeeze(0).cpu().numpy()

            attention_records.append({"video": video_name, "weights": w_np})

            plot_video_attention(inputs.cpu(), w_np, video_name, attn_dir)

    analyze_attention_distribution(attention_records)


def run_inference_transformer_all_rope(model, loader, device, run_name, base_out_dir):
    model.eval()
    attention_records = []
    dataset_samples = loader.dataset.data
    attn_dir = os.path.join(base_out_dir, f"attention_vis_{run_name}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Transformer Inference (RoPE, multi-class)")):
            inputs = batch[0].squeeze(0).to(device)
            frame_positions = batch[3].to(device)  # keep [1, num_frames], matches cls_position's [1,1] for cat
            video_name = dataset_samples[i]['vid_name']

            _, weights = model(inputs, frame_positions)
            w_np = weights.squeeze(0).cpu().numpy()

            attention_records.append({"video": video_name, "weights": w_np})

            plot_video_attention(inputs.cpu(), w_np, video_name, attn_dir)

    analyze_attention_distribution(attention_records)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    for split in args.splits:
        # Load Video Split
        json_path = os.path.join(args.annotations_path, f"split_{split}", f"mil_val_{args.dataset_name}.json")
        if args.use_rope:
            val_ds = MILVideoDatasetRope(
                json_path=json_path,
                num_frames=32,
                transform=val_trans,
                search_dir_paths=[args.videos_path],
            )
        else:
            val_ds = MILVideoDataset(
                json_path=json_path,
                num_frames=32,
                transform=val_trans,
                search_dir_paths=[args.videos_path],
            )
        if args.limit:
            # Grab one video per grade (in dataset order) for a representative,
            # fast-to-render sample instead of the full val set.
            seen_grades = set()
            subset = []
            for item in val_ds.data:
                if item['grade'] not in seen_grades:
                    subset.append(item)
                    seen_grades.add(item['grade'])
                if len(subset) >= args.limit:
                    break
            val_ds.data = subset

        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        run_name = f"split_{split}"
        print(f"\n>>> Analyzing Attention for Run: {run_name}")

        if args.use_rope:
            trans_model = DinoSelfAttentionRope(
                full_checkpoint_path=args.trans_paths[split - 1],
                num_blocks=args.num_blocks,
                use_rope=True,
                rope_base=args.rope_base,
                num_classes=1 if args.binary_classification else 5,
            ).to(device)
            if args.binary_classification:
                raise NotImplementedError("RoPE attention viz currently only wired for 5-class.")
            run_inference_transformer_all_rope(trans_model, val_loader, device, run_name, args.output_dir)
        elif args.binary_classification:
            trans_model = DinoSelfAttention(full_checkpoint_path=args.trans_paths[split - 1]).to(device)
            run_inference_transformer(trans_model, val_loader, device, run_name, args.output_dir)
        else:
            trans_model = DinoSelfAttention(full_checkpoint_path=args.trans_paths[split - 1], num_classes=5).to(device)
            run_inference_transformer_all(trans_model, val_loader, device, run_name, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--trans_paths', nargs='+', required=True, help="List of 5 Transformer checkpoint paths")
    parser.add_argument('--dataset_name', type=str, default="2025")
    parser.add_argument('--annotations_path', type=str, default="data/stratified_splits")
    parser.add_argument('--videos_path', type=str, default="data/ensemble_results_paxos2025/cleaned_videos")
    parser.add_argument('--splits', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--limit', type=int, default=None, help="Only render this many videos per split (one per grade where possible).")
    parser.add_argument('--use_rope', action='store_true')
    parser.add_argument('--rope_base', type=float, default=1000)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default="transformer_attention_analysis")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--binary_classification', action='store_true')

    args = parser.parse_args()
    main(args)