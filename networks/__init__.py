import segmentation_models_pytorch as smp

from .dual_model import DualUNetFormerWrapper, UniversalDualWrapper
from .M3LSNet import M3LSNet
from .Mask2Former import HF_Mask2Former, HF_Mask2Former_ConvNeXt
from .OCRNet import SMP_OCRNet
from .OneFormer import HF_OneFormer, HF_OneFormer_ConvNeXt
from .UnetFormer import HF_UNetFormer


def get_model(model_name, in_channels=7, num_classes=2):
    """
    get model instance

    params:
    - model_name:
        [
            # --- CNN Group (ResNeSt50 / ConvNeXttiny) ---
            'deeplabv3p_resnest50', 'unetplusplus_resnest50',
            'manet_resnest50', 'pspnet_resnest50',
            'upernet_convnexttiny', 'fpn_convnexttiny',

            # --- Transformer Group (MiT-B2 / Swin-Tiny) ---
            'deeplabv3p_mitb2', 'segformer_mitb2',
            'manet_swin', 'upernet_swin', 'unetplusplus_swin',

            # --- Special/Custom Models ---
            'ocrnet_hrnet_w48', 'm3lsnet',
            'mask2former-swin', 'oneformer-swin'
        ]
    - in_channels
    - num_classes

    返回:
    - model
    """
    requested_model_name = model_name
    # PMDFNet is the current paper-facing name.  Historical
    # ``dual_segformer_*`` identifiers remain valid for old checkpoints and
    # reproduction scripts.
    if model_name.startswith("pmdfnet_"):
        model_name = "dual_segformer_" + model_name[len("pmdfnet_") :]

    print(f"loading {requested_model_name}...")

    if model_name == "deeplabv3p_resnest50":
        return smp.DeepLabV3Plus(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "deeplabv3p_convnexttiny":
        return smp.DeepLabV3Plus(
            encoder_name="tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "deeplabv3p_mitb2":
        return smp.DeepLabV3Plus(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    elif model_name == "unet_resnest50":
        return smp.Unet(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "unet_resnet50":
        return smp.Unet(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "unet_convnexttiny":
        return smp.Unet(
            encoder_name="tu-convnext_tiny",
            encoder_weights="imagenet",
            encoder_depth=4,
            decoder_channels=(256, 128, 64, 32),
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "unet_convnexttinyv2":
        return smp.Unet(
            encoder_name="tu-convnextv2_tiny",
            encoder_weights="imagenet",
            encoder_depth=4,
            decoder_channels=(256, 128, 64, 32),
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "unet_mitb2":
        return smp.Unet(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            encoder_depth=4,
            decoder_channels=(256, 128, 64, 32),
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "unet_pvtb2":
        return smp.Unet(
            encoder_name="tu-pvt_v2_b2",
            encoder_weights="imagenet",
            encoder_depth=4,
            decoder_channels=(256, 128, 64, 32),
            in_channels=in_channels,
            classes=num_classes,
        )

    elif model_name == "unetplusplus_resnest50":
        return smp.UnetPlusPlus(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    elif model_name == "manet_resnest50":
        return smp.MAnet(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    elif model_name == "manet_convnexttiny":
        return smp.MAnet(
            encoder_name="tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "manet_mitb2":
        return smp.MAnet(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            encoder_depth=4,
            decoder_channels=(256, 128, 64, 32),
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "manet_pvtb2":
        return smp.MAnet(
            encoder_name="tu-pvt_v2_b2",
            encoder_weights="imagenet",
            encoder_depth=4,
            decoder_channels=(256, 128, 64, 32),
            in_channels=in_channels,
            classes=num_classes,
        )

    elif model_name == "pspnet_resnest50":
        return smp.PSPNet(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    elif model_name == "pspnet_convnexttiny":
        return smp.PSPNet(
            encoder_name="tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    elif model_name == "pspnet_mitb2":
        return smp.PSPNet(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            encoder_depth=4,
            decoder_channels=(256, 128, 64, 32),
            upsampling=16,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "pspnet_pvtb2":
        return smp.PSPNet(
            encoder_name="tu-pvt_v2_b2",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    elif model_name == "upernet_resnest50":
        return smp.UPerNet(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "upernet_convnexttiny":
        return smp.UPerNet(
            encoder_name="tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "upernet_convnextsmall":
        return smp.UPerNet(
            encoder_name="tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "upernet_convnexttinyv2":
        return smp.UPerNet(
            encoder_name="tu-convnextv2_tiny",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "upernet_mitb2":
        return smp.UPerNet(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            decoder_channels=256,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "upernet_pvtb2":
        return smp.UPerNet(
            encoder_name="tu-pvt_v2_b2",
            encoder_weights="imagenet",
            decoder_channels=256,
            in_channels=in_channels,
            classes=num_classes,
        )

    elif model_name == "fpn_resnest50":
        return smp.FPN(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "fpn_convnexttiny":
        return smp.FPN(
            encoder_name="tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "fpn_mitb2":
        return smp.FPN(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "fpn_pvtb2":
        return smp.FPN(
            encoder_name="tu-pvt_v2_b2",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    elif model_name == "segformer_resnest50":
        return smp.Segformer(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "segformer_convnexttiny":
        return smp.Segformer(
            encoder_name="tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "segformer_convnexttinyv2":
        return smp.Segformer(
            encoder_name="tu-convnextv2_tiny",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "segformer_convnextsmall":
        return smp.Segformer(
            encoder_name="tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "segformer_convnextbase":
        return smp.Segformer(
            encoder_name="tu-convnext_base",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "segformer_convnextlarge":
        return smp.Segformer(
            encoder_name="tu-convnext_large",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "segformer_mitb2":
        return smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "segformer_pvtb2":
        return smp.Segformer(
            encoder_name="tu-pvt_v2_b2",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    elif model_name == "ocrnet_hrnet_w48":
        return SMP_OCRNet(
            encoder_name="tu-hrnet_w48", in_channels=in_channels, classes=num_classes
        )
    elif model_name == "ocrnet_hrnet_w18":
        return SMP_OCRNet(
            encoder_name="tu-hrnet_w18", in_channels=in_channels, classes=num_classes
        )
    elif model_name == "mask2former-swintiny":
        return HF_Mask2Former(
            backbone="swin-tiny", in_channels=in_channels, classes=num_classes
        )
    elif model_name == "oneformer-swintiny":
        return HF_OneFormer(
            backbone="swin-tiny", in_channels=in_channels, num_classes=num_classes
        )

    elif model_name == "m3lsnet":
        return M3LSNet(input_channels=in_channels, num_classes=num_classes)

    elif model_name == "unetformer_convnexttiny":
        return HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_tiny",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=in_channels,
            out_indices=(0, 1, 2, 3),
        )
    elif model_name == "oneformer-convnexttiny":
        return HF_OneFormer_ConvNeXt(
            in_channels=in_channels, num_classes=num_classes, backbone="convnext-tiny"
        )
    elif model_name == "mask2former-convnexttiny":
        return HF_Mask2Former_ConvNeXt(in_channels=in_channels, classes=num_classes)

    elif model_name == "dual_unet_convnexttiny_chv1_add":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )

        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")

    elif model_name == "dual_unet_convnexttiny_chv2_add":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )

        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")

    elif model_name == "dual_unet_convnexttiny_chv1_att":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )

        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="att")

    elif model_name == "dual_unet_convnexttiny_chv2_att":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )

        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="att")

    elif model_name == "dual_unet_convnexttiny_chv1_cat":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )

        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="cat")

    elif model_name == "dual_unet_convnexttiny_chv2_cat":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Unet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )

        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="cat")

    elif model_name == "dual_upernet_convnextsmall_chv1_add":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.UPerNet(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.UPerNet(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_upernet_convnextsmall_chv2_add":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.UPerNet(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.UPerNet(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_upernet_convnexttiny_chv1_add":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.UPerNet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.UPerNet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_upernet_convnexttiny_chv2_add":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.UPerNet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.UPerNet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_upernet_convnexttiny_chv2_cat":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.UPerNet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.UPerNet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="cat")
    elif model_name == "dual_upernet_convnexttiny_chv2_att":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.UPerNet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.UPerNet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="att")
    elif model_name == "dual_upernet_convnexttiny_chv2_moe":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.UPerNet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.UPerNet(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="moe")

    elif model_name == "dual_segformer_convnextsmall_chv1_add":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_segformer_convnextsmall_chv2_add":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_segformer_convnextsmall_chv3_add":
        ch1, ch2 = [0, 1, 2], [3, 4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")

    elif model_name == "dual_segformer_convnextsmall_chv1_cat":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="cat")
    elif model_name == "dual_segformer_convnextsmall_chv1_att":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="att")
    elif model_name == "dual_segformer_convnextsmall_chv1_moe":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_small",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="moe")

    elif model_name == "dual_segformer_convnextbase_chv1_add":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_base",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_base",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_segformer_convnextlarge_chv1_add":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_large",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_large",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")

    elif model_name == "dual_segformer_convnexttiny_chv1_add":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")

    elif model_name == "dual_segformer_convnexttiny_chv1_cat":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="cat")
    elif model_name == "dual_segformer_convnexttiny_chv1_att":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="att")
    elif model_name == "dual_segformer_convnexttiny_chv1_moe":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="moe")
    elif model_name == "dual_segformer_convnexttiny_chv1_moev2":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="moev2")

    elif model_name == "dual_segformer_convnextv2tiny_chv1_add":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnextv2_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnextv2_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_segformer_convnexttiny_chv2_add":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_segformer_convnexttiny_chv3_add":
        ch1, ch2 = [0, 1, 2], [3, 4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_segformer_convnexttiny_chv2_cat":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="cat")
    elif model_name == "dual_segformer_convnexttiny_chv2_att":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="att")
    elif model_name == "dual_segformer_convnexttiny_chv2_moe":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]
        main_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch1),
            classes=num_classes,
        )
        aux_m = smp.Segformer(
            "tu-convnext_tiny",
            encoder_weights="imagenet",
            in_channels=len(ch2),
            classes=num_classes,
        )
        return UniversalDualWrapper(main_m, aux_m, ch1, ch2, fusion_type="moe")

    elif model_name == "dual_unetformer_convnextsmall_chv1_add":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]

        main_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_small",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch1),
            out_indices=(0, 1, 2, 3),
        )

        aux_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_small",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch2),
            out_indices=(0, 1, 2, 3),
        )

        return DualUNetFormerWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_unetformer_convnextbase_chv1_add":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]

        main_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_base",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch1),
            out_indices=(0, 1, 2, 3),
        )

        aux_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_base",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch2),
            out_indices=(0, 1, 2, 3),
        )

        return DualUNetFormerWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_unetformer_convnextlarge_chv1_add":
        ch1, ch2 = [0, 1, 2, 3], [4, 5, 6]

        main_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_large",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch1),
            out_indices=(0, 1, 2, 3),
        )

        aux_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_large",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch2),
            out_indices=(0, 1, 2, 3),
        )

        return DualUNetFormerWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_unetformer_convnexttiny_chv2_add":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]

        main_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_tiny",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch1),
            out_indices=(0, 1, 2, 3),
        )

        aux_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_tiny",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch2),
            out_indices=(0, 1, 2, 3),
        )

        return DualUNetFormerWrapper(main_m, aux_m, ch1, ch2, fusion_type="add")
    elif model_name == "dual_unetformer_convnexttiny_chv2_cat":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]

        main_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_tiny",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch1),
            out_indices=(0, 1, 2, 3),
        )

        aux_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_tiny",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch2),
            out_indices=(0, 1, 2, 3),
        )

        return DualUNetFormerWrapper(main_m, aux_m, ch1, ch2, fusion_type="cat")
    elif model_name == "dual_unetformer_convnexttiny_chv2_att":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]

        main_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_tiny",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch1),
            out_indices=(0, 1, 2, 3),
        )

        aux_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_tiny",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch2),
            out_indices=(0, 1, 2, 3),
        )

        return DualUNetFormerWrapper(main_m, aux_m, ch1, ch2, fusion_type="att")
    elif model_name == "dual_unetformer_convnexttiny_chv2_moe":
        ch1, ch2 = [0, 1, 2], [4, 5, 6]

        main_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_tiny",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch1),
            out_indices=(0, 1, 2, 3),
        )

        aux_m = HF_UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="convnext_tiny",
            pretrained=True,
            window_size=8,
            num_classes=num_classes,
            in_channels=len(ch2),
            out_indices=(0, 1, 2, 3),
        )

        return DualUNetFormerWrapper(main_m, aux_m, ch1, ch2, fusion_type="moe")
    else:
        ValueError(f"Model {model_name} not found")
