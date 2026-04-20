import torch
from torchvision.datasets import CocoDetection

class PlantDocDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(PlantDocDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self._clean_data()

    def _clean_data(self):
        """
        Cleans the COCO dataset by checking for invalid bounding boxes 
        and removing images that have no valid bounding boxes left.
        """
        valid_ids = []
        removed_images = 0
        removed_boxes = 0

        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            valid_anns = []
            for ann in anns:
                if 'bbox' not in ann:
                    continue
                
                xmin, ymin, w, h = ann['bbox']
                # Check for invalid sizes
                if w <= 1.0 or h <= 1.0:
                    removed_boxes += 1
                    continue
                
                valid_anns.append(ann)
            
            # Keep the image if it has valid annotations
            if len(valid_anns) > 0:
                valid_ids.append(img_id)
            else:
                removed_images += 1
                
        self.ids = valid_ids
        print(f"Dataset Cleaning: Removed {removed_images} images with no annotations/valid boxes.")
        print(f"Dataset Cleaning: Ignored {removed_boxes} invalid bounding boxes.")

    def __getitem__(self, idx):
        # We must override this to format the data for DETR's loss function
        img, target_orig = super(PlantDocDataset, self).__getitem__(idx)
        # Handle case where the dataset acts on self.ids[idx]
        image_id = torch.tensor([self.ids[idx]])
        
        # DETR expects normalized [cx, cy, w, h] boxes
        w_orig, h_orig = img.size
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for obj in target_orig:
            xmin, ymin, w, h = obj['bbox']
            
            # Boundary checks (so boxes don't fall off the image)
            xmax = min(xmin + w, w_orig)
            ymax = min(ymin + h, h_orig)
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            
            w_valid = xmax - xmin
            h_valid = ymax - ymin
            
            if w_valid <= 1.0 or h_valid <= 1.0:
                continue

            # Convert to center x, center y
            cx = xmin + w_valid / 2
            cy = ymin + h_valid / 2
            
            # Normalize to 0-1
            boxes.append([cx / w_orig, cy / h_orig, w_valid / w_orig, h_valid / h_orig])
            labels.append(obj['category_id'])
            
            # Unnormalized area (standard for COCO map eval)
            areas.append(obj.get('area', w_valid * h_valid))
            iscrowd.append(obj.get('iscrowd', 0))
            
        # Handle cases where image has no annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': iscrowd
        }
        
        if self._transforms is not None:
            # Handle standard torchvision transforms that only take image
            try:
                img, target = self._transforms(img, target)
            except TypeError:
                img = self._transforms(img)
            
        return img, target

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))