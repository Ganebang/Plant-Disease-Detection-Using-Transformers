import sys
import math
import torch
from torch.amp import GradScaler, autocast # For 6GB VRAM efficiency
from torchvision.ops import box_convert

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, scaler, accumulation_steps=8, scheduler=None):
    model.train()
    criterion.train()
    
    optimizer.zero_grad()

    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = len(data_loader)

    for i, (images, targets) in enumerate(data_loader):
        images = torch.stack([img.to(device) for img in images])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 1. Forward Pass with Mixed Precision (Memory Saving)
        with autocast('cuda'):
            outputs = model(images)
            # [Explanation]: The loss for DETR uses Bipartite Matching
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_value = losses.item()
        
        # [Improvement] Safety Check for NaN/Inf Losses to avoid silently corrupting the model
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        # 2. Backward Pass (Scaling loss to fit in 16-bit)
        # Note: We divide by accumulation_steps for a "Virtual Batch Size"
        scaler.scale(losses / accumulation_steps).backward()

        # 3. Optimizer Step (Every 'N' steps to save VRAM)
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Step the inner-epoch scheduler if provided
            if scheduler is not None:
                scheduler.step()

        epoch_loss += loss_value
        epoch_acc += loss_dict.get('class_acc', torch.tensor(0.0)).item()

        if i % 10 == 0:
            print(f"Epoch {epoch + 1}, Iteration {i + 1}, Loss: {loss_value:.4f}, Acc: {loss_dict.get('class_acc', torch.tensor(0.0)).item():.4f}")

    # 4. Flush remaining gradients that didn't hit accumulation length evenly
    if len(data_loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    return float(epoch_loss / num_batches), float(epoch_acc / num_batches)

@torch.inference_mode()
def evaluate(model, criterion, data_loader, device):
    """
    Evaluates the model without tracking gradients and calculates 
    mAP (Mean Average Precision) using torchmetrics.
    """
    model.eval()
    criterion.eval()
    
    epoch_val_loss = 0.0
    epoch_val_acc = 0.0
    val_map_score = 0.0
    num_batches = len(data_loader)
    
    # Initialize torchmetrics mAP evaluator
    map_metric = None
    if HAS_TORCHMETRICS:
        map_metric = MeanAveragePrecision()
        map_metric.to(device)
    else:
        print("Notice: 'torchmetrics' is not installed. Skipping mAP calculation.")

    for i, (images, targets) in enumerate(data_loader):
        images_tensor = torch.stack([img.to(device) for img in images])
        targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images_tensor)
        loss_dict = criterion(outputs, targets_gpu)
        weight_dict_c = criterion.weight_dict
        losses_c = sum(loss_dict[k] * weight_dict_c[k] for k in loss_dict.keys() if k in weight_dict_c)
        
        epoch_val_loss += losses_c.item()
        epoch_val_acc += loss_dict.get('class_acc', torch.tensor(0.0)).item()
        
        # Format for mAP Evaluation if torchmetrics is available
        if map_metric is not None:
            # 1. Post-process model outputs
            out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
            prob = torch.nn.functional.softmax(out_logits, -1)
            scores, labels = prob[..., :-1].max(-1) # Strip background class
            
            # Convert DETR's normal [cx, cy, w, h] to [xmin, ymin, xmax, ymax]
            boxes = box_convert(out_bbox, 'cxcywh', 'xyxy')
            
            # Scale boxes up to the tensor dimension
            _, _, h, w = images_tensor.shape
            scale_fct = torch.tensor([w, h, w, h], dtype=torch.float32, device=device)
            boxes = boxes * scale_fct[None, None, :]
            
            preds = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
            
            # 2. Process targets (they are also normalized cxcywh currently!)
            processed_targets = []
            for t in targets_gpu:
                t_boxes = box_convert(t['boxes'], 'cxcywh', 'xyxy')
                t_boxes = t_boxes * scale_fct[None, :]
                processed_targets.append({
                    'boxes': t_boxes,
                    'labels': t['labels'],
                })
                
            map_metric.update(preds, processed_targets)
            
    avg_val_loss = float(epoch_val_loss / num_batches)
    avg_val_acc = float(epoch_val_acc / num_batches)
    
    if map_metric is not None:
        metrics = map_metric.compute()
        print(f"Validation mAP@50-95: {metrics['map']:.4f} | mAP@50: {metrics['map_50']:.4f}")
        val_map_score = metrics['map'].item()
        map_metric.reset()
        
    return avg_val_loss, avg_val_acc, val_map_score
