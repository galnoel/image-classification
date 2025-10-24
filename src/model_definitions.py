# src/model_definitions.py
import torchvision.models as models
import timm

# This dictionary is now the single source of truth for all models.
# It maps the model's string name to its creation function.
AVAILABLE_MODELS = {
    "efficientnet_b0": models.efficientnet_b0,
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "vit_b_16": models.vit_b_16,
    "convnext_tiny": models.convnext_tiny,
    "swin_t": models.swin_t,
    "maxvit_tiny": lambda weights: timm.create_model('maxvit_tiny_tf_224', pretrained=True if weights else False),
    "cvt_13": lambda weights: timm.create_model('cvt_13_224', pretrained=True if weights else False),
    "coat_lite_mini": lambda weights: timm.create_model('coat_lite_mini', pretrained=True if weights else False),
    "efficientformerv2_s0": lambda weights: timm.create_model('efficientformerv2_s0', pretrained=True if weights else False),
    "levit_192": lambda weights: timm.create_model('levit_192', pretrained=True if weights else False),
}