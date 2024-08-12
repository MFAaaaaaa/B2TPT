from PIL.Image import Image
import torch
from typing import Any, Union, List
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from modeling_orig import VisionTransformer, CONFIGS


try:
    from torchvision.transforms import InterpolationMode, Compose

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    config = "vit"
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=args.num_classes)
    # model.load_from(np.load(args.pretrained_dir))
    # model.load_state_dict(torch.load(args.pretrained_dir))
    model.to(device)
    # num_params = count_parameters(model)
    #
    # logger.info("{}".format(config))
    # logger.info("Training parameters %s", args)
    # logger.info("Total Parameter: \t%2.1fM" % num_params)
    # print(num_params)
    return model, _transform(model.input_resolution.item())