import segmentation_models_pytorch as smp


def get_deeplab_model(num_classes=1, encoder_name="mobilenet_v2", encoder_weights="imagenet"):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    return model
