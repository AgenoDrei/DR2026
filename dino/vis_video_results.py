import numpy as np
import torch
import pandas as pd
import os
import json
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader

from data.dataset import MILVideoDataset, crop_black_borders
from dino.utils import make_val_transform_dino
from misc.training import add_zoom_args, zoom_settings
from dino.train_dino_transformer import DinoSelfAttention
from dino.train_dino_classifier import DinoClassifier
from convnext.model import ConvNextSelfAttention


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
class FrameDataset(Dataset):
    def __init__(self, json_path, img_dirs, transform=None, crop_borders=False):
        self.crop_borders = crop_borders
        with open(json_path, 'r') as f:
            label_data = json.load(f)

        self.filename_to_path = {}
        for img_dir in img_dirs:
            for fname in os.listdir(img_dir):
                full_path = os.path.join(img_dir, fname)
                if fname in self.filename_to_path:
                    continue
                self.filename_to_path[fname] = full_path

        self.image_paths = []
        self.image_names = []
        self.grades = []
        self.labels = []
        missing = []

        for img_name, grade in label_data.items():
            if img_name not in self.filename_to_path:
                missing.append(img_name)
                continue

            self.image_paths.append(self.filename_to_path[img_name])
            self.image_names.append(img_name)
            self.grades.append(grade)
            self.labels.append(1 if grade >= 2 else 0)

        if missing:
            print(f"WARNING: {len(missing)} image(s) from {json_path} were not "
                  f"found in any of img_dirs and were skipped.")

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.crop_borders:
            img = Image.fromarray(crop_black_borders(np.asarray(img)))
        if self.transform:
            img = self.transform(img)

        return (
            img,
            torch.tensor(self.grades[idx], dtype=torch.int64),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            self.image_names[idx]
        )


def match_img_to_video(img_name, vid_names):
    """Matches a frame image name back to its corresponding video key in the dataset."""
    for vid_name in vid_names:
        base_name = vid_name.replace('.mp4', '').replace('.MOV', '')
        if base_name in img_name:
            return vid_name
    return None


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def to_probs(logits, num_classes):
    """Logits -> probabilities of shape (N, C). Binary: sigmoid with C=1. Multi-class: softmax."""
    logits = logits.float()  # fp16 under autocast
    if num_classes == 1:
        return torch.sigmoid(logits.reshape(-1, 1))
    return torch.softmax(logits.reshape(-1, num_classes), dim=1)


def make_result(label, grade, probs):
    """Builds a result dict from a 1D probability tensor (length 1 = binary, else multi-class)."""
    result = {"label": int(label), "grade": int(grade)}
    if probs.numel() == 1:
        result["prob"] = probs[0].item()
    else:
        result["pred_grade"] = int(torch.argmax(probs).item())
        for c in range(probs.numel()):
            result[f"prob_{c}"] = probs[c].item()
    return result


def autocast(device, amp):
    """fp16 autocast on CUDA (as in validate_extended_trans), ~3x faster than fp32 for ViT-L at 512px."""
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda")


def cache_bags(loader, amp):
    """Decodes every val video once per split: [(vid_name, bag [T, C, H, W] on CPU, label, grade)], shared by all
    video models (val sampling is deterministic). fp16 under amp (~50MB per 32-frame bag)."""
    dtype = torch.float16 if amp else torch.float32
    names = [sample['vid_name'] for sample in loader.dataset.data]
    return [(names[i], batch[0].squeeze(0).to(dtype), batch[1].item(), batch[2].item())
            for i, batch in enumerate(tqdm(loader, desc="Decoding videos"))]


# ----------------------------------------------------------------------------
# Model types: each has a builder and an inference function.
#   builder(path, num_classes, device)                      -> model
#   infer(model, data, device, num_classes, vid_names, amp) -> {video_name: result_dict}
# "data" says what the model consumes: "video" (cached bags, see cache_bags) or "frame" (frame DataLoader).
# To support a new model type, write both functions and add it to MODEL_REGISTRY.
# ----------------------------------------------------------------------------
def _check_num_classes(model, num_classes, path):
    model_classes = model.num_classes if hasattr(model, "num_classes") else model.head[-1].out_features
    if model_classes != num_classes:
        raise ValueError(f"{path} has {model_classes} output(s), expected {num_classes}.")


def build_transformer(path, num_classes, device):
    model = DinoSelfAttention.from_checkpoint(path)
    _check_num_classes(model, num_classes, path)
    return model.to(device)


def build_convnext_transformer(path, num_classes, device):
    model = ConvNextSelfAttention.from_checkpoint(path)
    _check_num_classes(model, num_classes, path)
    return model.to(device)


def infer_transformer(model, bags, device, num_classes, vid_names, amp):
    model.eval()
    results = {}

    with torch.no_grad(), autocast(device, amp):
        for video_name, inputs, label, grade in tqdm(bags, desc="Transformer Inference"):
            logits, _ = model(inputs.to(device))
            probs = to_probs(logits, num_classes)[0]

            results[video_name] = make_result(label, grade, probs)
    return results


def build_classifier(path, num_classes, device):
    model = DinoClassifier.from_checkpoint(path)
    _check_num_classes(model, num_classes, path)
    return model.to(device)


def infer_classifier_frames(model, loader, device, num_classes, vid_names, amp):
    """Classifier on the manually selected, pre-extracted frames.
    Predicts per frame and averages probabilities per video."""
    model.eval()
    video_probs = {v: [] for v in vid_names}
    video_labels = {}
    video_grades = {}

    with torch.no_grad(), autocast(device, amp):
        for batch in tqdm(loader, desc="Classifier Frames Inference"):
            inputs, grades, labels, img_names = batch
            inputs = inputs.to(device)

            probs = to_probs(model(inputs), num_classes).cpu()

            for i, img_name in enumerate(img_names):
                v_name = match_img_to_video(img_name, vid_names)
                if v_name:
                    video_probs[v_name].append(probs[i])
                    video_labels[v_name] = int(labels[i].item())
                    video_grades[v_name] = int(grades[i].item())

    results = {}
    for v_name in vid_names:
        if video_probs[v_name]:
            avg_prob = torch.stack(video_probs[v_name]).mean(dim=0)
            results[v_name] = make_result(video_labels[v_name], video_grades[v_name], avg_prob)
        else:
            print(f"Warning: No frames matched for video {v_name}")

    return results


def infer_classifier_video(model, bags, device, num_classes, vid_names, amp):
    """Classifier on the same 32 frames the transformer gets.
    Predicts per frame and averages probabilities per video."""
    model.eval()
    results = {}

    with torch.no_grad(), autocast(device, amp):
        for video_name, inputs, label, grade in tqdm(bags, desc="Classifier Video-Frames Inference"):
            probs = to_probs(model(inputs.to(device)), num_classes).mean(dim=0)  # inputs: (32, C, H, W)
            results[video_name] = make_result(label, grade, probs)

    return results


MODEL_REGISTRY = {
    "transformer":      {"data": "video", "build": build_transformer, "infer": infer_transformer},
    "convnext_transformer": {"data": "video", "build": build_convnext_transformer, "infer": infer_transformer},
    "classifier":       {"data": "frame", "build": build_classifier,  "infer": infer_classifier_frames},
    "classifier_video": {"data": "video", "build": build_classifier,  "infer": infer_classifier_video},
}


def average_results(passes):
    """Averages the probabilities of several inference passes ({video: result_dict}) and recomputes pred_grade."""
    if len(passes) == 1:
        return passes[0]
    results = {}
    for v, r in passes[0].items():
        prob_keys = [k for k in r if k == "prob" or k.startswith("prob_")]
        probs = torch.tensor([[p[v][k] for k in prob_keys] for p in passes]).mean(dim=0)
        results[v] = make_result(r["label"], r["grade"], probs)
    return results


# ----------------------------------------------------------------------------
# Output formatting
# ----------------------------------------------------------------------------
def format_model_columns(results, name, run_name, num_classes):
    """Turns one model's results into a DataFrame with model-specific column names.
    Keeps label/grade (merged later) plus renamed prediction columns."""
    df = pd.DataFrame.from_dict(results, orient='index')
    if num_classes == 1:
        df = df.rename(columns={"prob": f"{name}_{run_name}"})
        pred_cols = [f"{name}_{run_name}"]
    else:
        rename = {f"prob_{c}": f"{name}_prob_{c}" for c in range(num_classes)}
        rename["pred_grade"] = f"{name}_pred_grade_{run_name}"
        df = df.rename(columns=rename)
        pred_cols = [f"{name}_pred_grade_{run_name}"] + [f"{name}_prob_{c}" for c in range(num_classes)]
    return df, pred_cols


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_models(model_args, splits):
    """Parses repeated `--model NAME TYPE PATH...` into a list of dicts and validates it."""
    models = []
    seen = set()
    for spec in model_args:
        if len(spec) < 3:
            raise ValueError(f"--model needs NAME TYPE PATH [PATH ...], got: {spec}")
        name, mtype, paths = spec[0], spec[1], spec[2:]

        if mtype not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type '{mtype}' for '{name}'. "
                             f"Available: {list(MODEL_REGISTRY)}")
        if name in seen:
            raise ValueError(f"Duplicate model name '{name}'.")
        seen.add(name)
        if len(paths) != len(splits):
            raise ValueError(f"Model '{name}' has {len(paths)} path(s) but {len(splits)} split(s) "
                             f"were requested ({splits}).")
        models.append({"name": name, "type": mtype, "paths": paths})
    return models


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs, args.clahe, zoom=zoom_settings(args)[0])

    num_classes = 1 if args.binary_classification else args.num_classes
    models = parse_models(args.model, args.splits)
    needs_frames = any(MODEL_REGISTRY[m["type"]]["data"] == "frame" for m in models)

    for split_idx, split in enumerate(args.splits):
        run_name = f"split_{split}"
        csv_path = os.path.join(args.output_dir, f"predictions_{run_name}_{args.dataset_name}.csv")
        print(f"\n>>> Evaluating Run: {run_name}")

        # Video split (always needed: transformer / classifier_video input and video names).
        # --eval_bags K > 1: K seeded random-segment bags per video (test-time averaging), else one uniform bag.
        val_ds = MILVideoDataset(
            json_path=os.path.join(args.annotations_path, run_name, f"mil_val_{args.dataset_name}.json"),
            num_frames=args.eval_frames,
            transform=val_trans,
            search_dir_paths=[args.videos_path],
            crop_borders=args.crop_borders,
            random_segment_sample=args.eval_bags > 1,
        )
        vid_names = [sample['vid_name'] for sample in val_ds.data]
        bag_passes = []
        for k in range(args.eval_bags):
            loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4,
                                generator=torch.Generator().manual_seed(args.eval_seed + k))  # seeds numpy in the workers
            bag_passes.append(cache_bags(loader, not args.fp32))
        loaders = {"video": bag_passes}

        # Frame split (only if a frame-based model is requested)
        if needs_frames:
            frame_json_path = os.path.join(args.frame_annotations_path, run_name,
                                           f"frame_val_{args.dataset_name}.json")
            frame_ds = FrameDataset(
                json_path=frame_json_path,
                img_dirs=[args.frame_img_dir],
                transform=val_trans,
                crop_borders=args.crop_borders,
            )
            loaders["frame"] = DataLoader(frame_ds, batch_size=32, shuffle=False, num_workers=4)

        base_df = None
        pred_columns = []

        for m in models:
            spec = MODEL_REGISTRY[m["type"]]
            print(f"--- {m['name']} ({m['type']}): {m['paths'][split_idx]}")

            model = spec["build"](m["paths"][split_idx], num_classes, device)
            if spec["data"] == "video":
                results = average_results([spec["infer"](model, bags, device, num_classes, vid_names, not args.fp32)
                                           for bags in loaders["video"]])
            else:
                results = spec["infer"](model, loaders["frame"], device, num_classes, vid_names, not args.fp32)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            df, pred_cols = format_model_columns(results, m["name"], run_name, num_classes)
            pred_columns.extend(pred_cols)

            # label/grade from every model: a frame-based model may miss videos without matched frames
            if base_df is None:
                base_df = df[["label", "grade"] + pred_cols]
            else:
                base_df = base_df.join(df[pred_cols], how="outer")
                base_df[["label", "grade"]] = base_df[["label", "grade"]].combine_first(df[["label", "grade"]])

        base_df.index.name = 'video'
        base_df = base_df[["label", "grade"] + pred_columns]
        base_df.to_csv(csv_path)
        print(f"Saved {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', action='append', nargs='+', required=True,
        metavar=('NAME', 'TYPE_AND_PATHS'),
        help=("Repeatable. Format: --model NAME TYPE PATH_1 [PATH_2 ...] with one checkpoint path per "
              f"split in --splits. TYPE is one of {list(MODEL_REGISTRY)}.")
    )
    parser.add_argument('--splits', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help="Which splits to evaluate; each --model needs one path per split.")

    # Frame dataset arguments
    parser.add_argument('--frame_annotations_path', type=str, default="data/stratified_splits")
    parser.add_argument('--frame_img_dir', default="data/2024_Paxos_Frames/cropped_matched_frames")

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--annotations_path', type=str, default="data/stratified_splits")
    parser.add_argument('--videos_path', type=str, default="data/ensemble_results_paxos2020/cleaned_videos")
    parser.add_argument('--output_dir', type=str, default="multi_eval")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=5, help="Used when not --binary_classification.")
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--no_clahe', dest='clahe', action='store_false',
                        help='--complex_augs without CLAHE (models trained with --no_clahe).')
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--crop_borders', action='store_true', default=True,
                        help='Trim black frame borders for all models (default; must match training).')
    parser.add_argument('--no_crop_borders', dest='crop_borders', action='store_false',
                        help='Keep black borders (models trained before 2026-11-01 / with --no_crop_borders).')
    add_zoom_args(parser)  # default center zoom 1.1 (runs since 2026-11-12); --no_zoom for earlier runs
    parser.add_argument('--eval_frames', type=int, default=32, help="Frames per bag for the video models.")
    parser.add_argument('--eval_bags', type=int, default=1,
                        help="> 1: average K seeded random-segment bags per video instead of one uniform bag.")
    parser.add_argument('--eval_seed', type=int, default=0, help="Seed of the first --eval_bags bag (bag k: seed + k).")
    parser.add_argument('--fp32', action='store_true',
                        help="Disable fp16 autocast (default since 2026-10-29; earlier prediction CSVs used fp32).")

    args = parser.parse_args()
    main(args)
