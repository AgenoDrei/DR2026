import torch
import numpy as np
from tqdm import tqdm
from misc.utils import binary_metrics, multiclass_metrics

def train_one_epoch_trans(model, loader, criterion, optimizer, binary_classification, device, scaler=None, entropy_weight=0.0, distance_scale_weight=0.0):
    """
    Trains one epoch of the Transformer sequence aggregator model.

    Uses autocast + a GradScaler when `scaler` is given (needed to fit the
    unfrozen ViT-L backbone's activations over a frame bag in 24GB).

    entropy_weight > 0 adds an entropy-minimization term on the CLS->frame
    attention weights: H(w) = -sum(w_i * log(w_i)), added to the loss so
    gradient descent pushes attention toward a peakier (lower-entropy, more
    selective) distribution over frames instead of a diffuse/uniform one.

    distance_scale_weight > 0 rescales the classification loss itself by
    1 + distance_scale_weight * |pred_grade - true_grade| (pred_grade taken
    as a detached argmax, so it only reweights the existing CE gradient
    rather than opening a new gradient path through the argmax). A video
    graded 0 but predicted 4 gets a much larger loss than one predicted 1,
    unlike plain CE which only looks at the probability mass on the true
    class. Only meaningful for 5-class (ignored when binary_classification,
    where the two ordinal classes make distance-scaling redundant with CE).
    """
    model.train()
    total_loss = 0
    total_entropy = 0
    use_amp = scaler is not None and device.type == "cuda"

    for sequence, label, grade in tqdm(loader, desc="Training"):
        # sequence shape from loader: [1, num_frames, C, H, W] -> Squeeze out batch dim
        sequence = sequence.squeeze(0).to(device)
        if binary_classification:
            target = label.to(device).float().unsqueeze(1)  # [1, 1]
        else:
            target = grade.to(device).long()  # [1]

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            # Model returns (logits, frame_attention_weights)
            logits, weights = model(sequence)
            loss = criterion(logits, target)

            if distance_scale_weight > 0 and not binary_classification:
                with torch.no_grad():
                    pred_grade = logits.argmax(dim=-1)
                    distance = (pred_grade - target).abs().float()
                    scale = 1.0 + distance_scale_weight * distance
                loss = loss * scale.squeeze()

            entropy = -(weights.float() * torch.log(weights.float() + 1e-12)).sum()
            if entropy_weight > 0:
                loss = loss + entropy_weight * entropy

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_entropy += entropy.item()

    return total_loss / len(loader), total_entropy / len(loader)


def train_one_epoch_trans_rope(model, loader, criterion, optimizer, binary_classification, device, scaler=None):
    """
    Trains one epoch of the Transformer sequence aggregator model (RoPE variant).

    Uses autocast + a GradScaler when `scaler` is given (needed to fit the
    unfrozen ViT-L backbone's activations over a 32-frame bag in 24GB).
    """
    model.train()
    total_loss = 0
    use_amp = scaler is not None and device.type == "cuda"

    for sequence, label, grade, frame_positions in tqdm(loader, desc="Training"):
        # sequence shape from loader: [1, num_frames, C, H, W] -> Squeeze out batch dim
        sequence = sequence.squeeze(0).to(device)
        if binary_classification:
            target = label.to(device).float().unsqueeze(1)  # [1, 1]
        else:
            target = grade.to(device).long()  # [1]

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            # Model returns (logits, frame_attention_weights)
            logits, _ = model(sequence, frame_positions)
            loss = criterion(logits, target)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_extended_trans(model, loader, criterion, binary_classification, device):
    """
    Extended validation loop for tracking sequential frame sequence metrics
    such as PR-AUC, ROC-AUC, F1, Precision, and Recall (binary), or
    QWK, macro F1/precision/recall, and per-class breakdowns (multiclass).
    """
    model.eval()
    running_loss = 0.0
    y_true, y_probs = [], []
    use_amp = device.type == "cuda"

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        for inputs, labels, grades in loader:
            inputs = inputs.squeeze(0).to(device)
            if binary_classification:
                target = labels.to(device).float().unsqueeze(1)
            else:
                target = grades.to(device).long()

            outputs = model(inputs)

            # Safely unpack the tuple output (logits, attention_weights)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits, target)
            running_loss += loss.item()

            if binary_classification:
                probs = torch.sigmoid(logits)          # (1,1)
            else:
                probs = torch.softmax(logits, dim=1)     # (1,5)

            y_probs.extend(probs.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    y_true = np.array(y_true).reshape(-1) if binary_classification else np.array(y_true)
    y_probs = np.array(y_probs)

    if binary_classification:
        y_pred = (y_probs.reshape(-1) > 0.5).astype(float)
        metrics = binary_metrics(y_true, y_pred, y_probs.reshape(-1))
    else:
        y_pred = y_probs.argmax(axis=1)
        metrics = multiclass_metrics(y_true, y_pred, y_probs, num_classes=5)

    metrics['loss'] = running_loss / len(loader)
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    return metrics


def validate_extended_trans_rope(model, loader, criterion, binary_classification, device):
    """
    Extended validation loop for tracking sequential frame sequence metrics
    such as PR-AUC, ROC-AUC, F1, Precision, and Recall (binary), or
    QWK, macro F1/precision/recall, and per-class breakdowns (multiclass)
    (RoPE variant).
    """
    model.eval()
    running_loss = 0.0
    y_true, y_probs = [], []
    use_amp = device.type == "cuda"

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        for inputs, labels, grades, frame_positions in loader:
            inputs = inputs.squeeze(0).to(device)
            if binary_classification:
                target = labels.to(device).float().unsqueeze(1)
            else:
                target = grades.to(device).long()

            outputs = model(inputs, frame_positions)

            # Safely unpack the tuple output (logits, attention_weights)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits, target)
            running_loss += loss.item()

            if binary_classification:
                probs = torch.sigmoid(logits)          # (1,1)
            else:
                probs = torch.softmax(logits, dim=1)     # (1,5)

            y_probs.extend(probs.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    y_true = np.array(y_true).reshape(-1) if binary_classification else np.array(y_true)
    y_probs = np.array(y_probs)

    if binary_classification:
        y_pred = (y_probs.reshape(-1) > 0.5).astype(float)
        metrics = binary_metrics(y_true, y_pred, y_probs.reshape(-1))
    else:
        y_pred = y_probs.argmax(axis=1)
        metrics = multiclass_metrics(y_true, y_pred, y_probs, num_classes=5)

    metrics['loss'] = running_loss / len(loader)
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    return metrics