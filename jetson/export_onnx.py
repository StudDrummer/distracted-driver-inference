import torch
import torch.nn as nn
from torchvision import models


class DriverActionClassifier(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(3 * 576, num_classes)  # matches your previous model

    def forward(self, image, face, hand):
        im = self.backbone(image).flatten(1)
        f = self.backbone(face).flatten(1)
        ha = self.backbone(hand).flatten(1)
        combined = torch.cat([im, f, ha], dim=1)
        return self.classifier(combined)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mobilenet = models.mobilenet_v3_small(weights=None)
mobilenet.classifier = nn.Identity()

model = DriverActionClassifier(backbone=mobilenet, num_classes=10).to(device)
model.load_state_dict(torch.load("../models/driver_action_deploy.pth", map_location=device))
model.eval()


full = torch.randn(1, 3, 224, 224).to(device)
face = torch.randn(1, 3, 224, 224).to(device)
hand = torch.randn(1, 3, 224, 224).to(device)


torch.onnx.export(
    model,
    (full, face, hand),
    "../models/driver_action.onnx",
    opset_version=11,
    input_names=['full','face','hand'],
    output_names=['logits'],
    dynamic_axes={
        'full': {0: 'batch_size'},
        'face': {0: 'batch_size'},
        'hand': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    },
    do_constant_folding=True,
    keep_initializers_as_inputs=True,

)

print("ONNX exported successfully to ../models/driver_action.onnx")
