import segmentation_models_pytorch as smp

def get_model(architecture="linknet", encoder="resnet34", encoder_weights="imagenet"):
    """
    Returns the segmentation model.
    D-LinkNet style is achieved via LinkNet with dilated blocks if modified, 
    but for now we use standard LinkNet from SMP.
    """
    if architecture.lower() == "linknet":
        model = smp.Linknet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
    elif architecture.lower() == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
    else:
        raise ValueError(f"Architecture {architecture} not supported.")
        
    return model
