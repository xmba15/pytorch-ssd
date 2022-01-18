#!/usr/bin/env python
import math
import torch
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd

"""
Ref:
- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
- [ssd 300 voc](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L313-L319)
- [ssd 300 coco config](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L313-L319)
"""


def _get_args():
    import argparse

    parser = argparse.ArgumentParser("estimate default box parameters")
    parser.add_argument(
        "--net_type",
        type=str,
        choices=["vgg16-ssd", "mb1-ssd", "mb1-ssd-lite", "mb2-ssd-lite"],
        default="vgg16-ssd",
    )
    parser.add_argument("--image_size", type=int, help="input image size", default=300)
    parser.add_argument("--min_ratio", type=int, default=20)  # percentage 20%
    parser.add_argument("--max_ratio", type=int, default=90)  # percentage 90%
    args = parser.parse_args()

    return args


def _get_ssd_engine(net_type):
    if net_type == "vgg16-ssd":
        return create_vgg_ssd
    elif net_type in ["mb1-ssd", "mb1-ssd-lite", "mb2-ssd-lite"]:
        return create_mobilenetv1_ssd
    else:
        raise ValueError("not supported net type {}".format(net_type))


def _get_aspect_ratios(net_type):
    if net_type == "vgg16-ssd":
        return [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    elif net_type in ["mb1-ssd", "mb1-ssd-lite", "mb2-ssd-lite"]:
        return [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    else:
        raise ValueError("not supported net type {}".format(net_type))


def _generate_spec_string(feature_maps, steps, min_sizes, max_sizes, net_type):
    output = ""
    for feature_map, step, min_size, max_size, aspect_ratio in zip(
        feature_maps, steps, min_sizes, max_sizes, _get_aspect_ratios(net_type)
    ):
        output += "SSDSpec({}, {}, SSDBoxSizes({}, {}), {}),\n".format(
            feature_map, step, min_size, max_size, aspect_ratio
        )
    return output


def main():
    args = _get_args()
    ssd = _get_ssd_engine(args.net_type)(num_classes=21)
    x = torch.randn(1, 3, args.image_size, args.image_size)
    feature_maps = ssd(x, get_feature_map_size=True)
    steps = [
        math.ceil(args.image_size * 1.0 / feature_map) for feature_map in feature_maps
    ]
    step = int(math.floor((args.max_ratio - args.min_ratio) / (len(feature_maps) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(args.min_ratio, args.max_ratio + 1, step):
        min_sizes.append(args.image_size * ratio / 100.0)
        max_sizes.append(args.image_size * (ratio + step) / 100.0)
    min_sizes = [args.image_size * (args.min_ratio / 2) / 100.0] + min_sizes
    max_sizes = [args.image_size * args.min_ratio / 100.0] + max_sizes
    print(
        _generate_spec_string(feature_maps, steps, min_sizes, max_sizes, args.net_type)
    )


if __name__ == "__main__":
    main()
