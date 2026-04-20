import torch
from torch import nn

class PlantDocDETR(nn.Module):
    def __init__(self, num_classes, num_queries=100):
        super().__init__()
        # Load the fully pretrained DETR model from Facebook Research via PyTorch Hub
        # This downloads ~150MB of weights on the first run.
        self.detr = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
        
        # The COCO pre-trained model outputs 91 classes. We need to replace the final
        # layer to output exactly our PlantDoc `num_classes` + 1 (for the background "no object" class).
        # This allows the powerful backbone and transformer to fine-tune to our classes.
        in_features = self.detr.class_embed.in_features
        self.detr.class_embed = nn.Linear(in_features, num_classes + 1)

    def forward(self, x):
        # Pass the processed batch down to the hub model
        # It perfectly returns {'pred_logits': logits, 'pred_boxes': boxes} format 
        # which seamlessly links up with your SetCriterion loss function!
        return self.detr(x)