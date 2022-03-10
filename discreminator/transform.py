import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transforms(image_size):
    train_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.1, p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ]
    )
    return train_transform


def valid_transforms(image_size):
    valid_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ]
    )
    return valid_transform
