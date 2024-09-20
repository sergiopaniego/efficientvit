# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from efficientvit.models.efficientvit import (
    SegHead,
    EfficientViTSeg,
    efficientvit_seg_b0,
    efficientvit_seg_b1,
    efficientvit_seg_b2,
    efficientvit_seg_b3,
    efficientvit_seg_l1,
    efficientvit_seg_l2,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file
from efficientvit.models.utils import build_kwargs_from_config

__all__ = ["create_seg_model"]


REGISTERED_SEG_MODEL = {
    "cityscapes_carla": {
        "b0": "assets/checkpoints/seg/cityscapes/b0.pt",
        "b1": "assets/checkpoints/seg/cityscapes/b1.pt",
        "b2": "assets/checkpoints/seg/cityscapes/b2.pt",
        "b3": "assets/checkpoints/seg/cityscapes/b3.pt",
        ################################################
        "l1": "assets/checkpoints/seg/cityscapes/l1.pt",
        "l2": "assets/checkpoints/seg/cityscapes/l2.pt",
        ################################################
        "qi_tcp": "example/checkpoint/2024-09-16_11-58-22_best_checkpoint.pt"
    },
    "cityscapes": {
        "b0": "assets/checkpoints/seg/cityscapes/b0.pt",
        "b1": "assets/checkpoints/seg/cityscapes/b1.pt",
        "b2": "assets/checkpoints/seg/cityscapes/b2.pt",
        "b3": "assets/checkpoints/seg/cityscapes/b3.pt",
        ################################################
        "l1": "assets/checkpoints/seg/cityscapes/l1.pt",
        "l2": "assets/checkpoints/seg/cityscapes/l2.pt",
    },
    "ade20k": {
        "b1": "assets/checkpoints/seg/ade20k/b1.pt",
        "b2": "assets/checkpoints/seg/ade20k/b2.pt",
        "b3": "assets/checkpoints/seg/ade20k/b3.pt",
        ################################################
        "l1": "assets/checkpoints/seg/ade20k/l1.pt",
        "l2": "assets/checkpoints/seg/ade20k/l2.pt",
    },
}


def create_seg_model(
    name: str, dataset: str, pretrained=True, weight_url= None, **kwargs
) -> EfficientViTSeg:
    model_dict = {
        "b0": efficientvit_seg_b0,
        "b1": efficientvit_seg_b1,
        "b2": efficientvit_seg_b2,
        "b3": efficientvit_seg_b3,
        #########################
        "l1": efficientvit_seg_l1,
        "l2": efficientvit_seg_l2,
        "qi_tcp": efficientvit_seg_l1,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](dataset=dataset, **kwargs)

        set_norm_eps(model, 1e-7)

    print(REGISTERED_SEG_MODEL)
    print('*********')
    print(REGISTERED_SEG_MODEL[dataset])
    print('*********')
    print(REGISTERED_SEG_MODEL[dataset].get(name, None))
    print('*********')
    print(dataset)
    print('*********')
    print(name)
    

    if pretrained:
        weight_url = weight_url or REGISTERED_SEG_MODEL[dataset].get(name, None)
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)

    print('------------------')
    print(model.backbone)
    print('------------------')
    print(model.head) 
    '''
    head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=256,
            head_depth=3,
            expand_ratio=1,
            middle_op="fmbconv",
            final_expand=None,
            n_classes=29, ######
            act_func="gelu",
            **build_kwargs_from_config(kwargs, SegHead),
        )

    model = EfficientViTSeg(model.backbone, head)
    '''

    return model
