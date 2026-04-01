import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models.resnet import BasicBlock, ResNet


class CIFARResNet18Dropout(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10:
    - conv1: 3x3 stride 1
    - maxpool removed
    - dropout before final linear classifier
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.2):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.feature_dim = backbone.fc.in_features
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward_classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.forward_classifier(features)
        if return_features:
            return logits, features
        return logits


class CIFARResNetDropout(nn.Module):
    """
    Generic CIFAR-style ResNet with configurable block depths.
    Example: [1, 1, 1, 1] gives a smaller ResNet-10-like model.
    """

    def __init__(self, layers, num_classes: int = 10, dropout_p: float = 0.2):
        super().__init__()
        backbone = ResNet(block=BasicBlock, layers=layers, num_classes=num_classes)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.feature_dim = backbone.fc.in_features
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward_classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.forward_classifier(features)
        if return_features:
            return logits, features
        return logits


class CIFARSmallCNNDropout(nn.Module):
    """Very small CNN for fast experimentation."""

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.hidden = nn.Linear(128 * 8 * 8, 256)
        self.feature_dim = 256
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.hidden(x))
        return x

    def forward_classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.forward_classifier(features)
        if return_features:
            return logits, features
        return logits


class CIFARVGG16Dropout(nn.Module):
    """
    VGG16 backbone adapted for small-image active learning benchmarks.
    Uses features -> adaptive pool -> linear classifier with dropout.
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.2):
        super().__init__()
        backbone = tv_models.vgg16(weights=None)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward_classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.forward_classifier(features)
        if return_features:
            return logits, features
        return logits


def build_model(model_name: str = "small_cnn", num_classes: int = 10, dropout_p: float = 0.2) -> nn.Module:
    if model_name == "resnet18":
        return CIFARResNet18Dropout(num_classes=num_classes, dropout_p=dropout_p)
    if model_name == "resnet10":
        return CIFARResNetDropout(layers=[1, 1, 1, 1], num_classes=num_classes, dropout_p=dropout_p)
    if model_name == "vgg16":
        return CIFARVGG16Dropout(num_classes=num_classes, dropout_p=dropout_p)
    if model_name == "small_cnn":
        return CIFARSmallCNNDropout(num_classes=num_classes, dropout_p=dropout_p)
    raise ValueError(f"Unsupported model: {model_name}")


def enable_dropout_layers_only(model: nn.Module) -> None:
    """Keeps model in eval mode except dropout layers for MC-dropout style passes."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
