from typing import Tuple
from tinygrad.tensor import Tensor, Device
from tinygrad.nn import Conv2d, BatchNorm2d


def channel_shuffle(x: Tensor) -> Tuple[Tensor, Tensor]:
    b, c, h, w = x.shape
    assert c % 4 == 0
    x = x.reshape(b * c // 2, w, h * 2).permute(1, 0, 2)
    x = x.reshape(2, -1, c // 2, h, w)
    return x[0], x[1]


class ShuffleV2Block:
    def __init__(self, inp: int, outp: int, c_mid: int, kernel_size: int, stride: int):
        assert stride in [1, 2]
        self.stride = stride
        self.inp, self.c_mid = inp, c_mid
        pad = kernel_size // 2
        out = outp - inp

        # pw
        self.cv1 = Conv2d(inp, c_mid, 1, 1, 0, bias=False)
        self.bn1 = BatchNorm2d(c_mid)

        # dw
        self.cv2 = Conv2d(
            c_mid, c_mid, kernel_size, stride, pad, groups=c_mid, bias=False
        )
        self.bn2 = BatchNorm2d(c_mid)

        # pw-linear
        self.cv3 = Conv2d(c_mid, out, 1, 1, 0, bias=False)
        self.bn3 = BatchNorm2d(out)

        if stride == 2:
            # dw
            self.cv4 = Conv2d(
                inp, inp, kernel_size, stride, pad, groups=inp, bias=False
            )
            self.bn4 = BatchNorm2d(inp)

            # pw-linear
            self.cv5 = Conv2d(inp, inp, 1, 1, 0, bias=False)
            self.bn5 = BatchNorm2d(inp)

    def __call__(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x_proj, x = channel_shuffle(x)
            x = self.bn1(self.cv1(x)).relu()
            x = self.bn2(self.cv2(x))
            x = self.bn3(self.cv3(x)).relu()
            return x_proj.cat(x, dim=1)
        elif self.stride == 2:
            x_proj = self.bn4(self.cv4(x))
            x_proj = self.bn5(self.cv5(x_proj)).relu()
            x = self.bn1(self.cv1(x)).relu()
            x = self.bn2(self.cv2(x))
            x = self.bn3(self.cv3(x)).relu()
            return x_proj.cat(x, dim=1)


class ShuffleNetV2:
    def __init__(self):
        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [24, 48, 96, 192]

        self.stage1 = [
            Conv2d(3, self.stage_out_channels[0], 3, 2, 1, bias=False),
            BatchNorm2d(self.stage_out_channels[0]),
        ]

        self.stage2 = [
            ShuffleV2Block(
                self.stage_out_channels[0],
                self.stage_out_channels[1],
                self.stage_out_channels[1] // 2,
                kernel_size=3,
                stride=2,
            )
        ] + [
            ShuffleV2Block(
                self.stage_out_channels[1] // 2,
                self.stage_out_channels[1],
                self.stage_out_channels[1] // 2,
                kernel_size=3,
                stride=1,
            )
            for _ in range(self.stage_repeats[0] - 1)
        ]

        self.stage3 = [
            ShuffleV2Block(
                self.stage_out_channels[1],
                self.stage_out_channels[2],
                self.stage_out_channels[2] // 2,
                kernel_size=3,
                stride=2,
            )
        ] + [
            ShuffleV2Block(
                self.stage_out_channels[2] // 2,
                self.stage_out_channels[2],
                self.stage_out_channels[2] // 2,
                kernel_size=3,
                stride=1,
            )
            for _ in range(self.stage_repeats[1] - 1)
        ]

        self.stage4 = [
            ShuffleV2Block(
                self.stage_out_channels[2],
                self.stage_out_channels[3],
                self.stage_out_channels[3] // 2,
                kernel_size=3,
                stride=2,
            )
        ] + [
            ShuffleV2Block(
                self.stage_out_channels[3] // 2,
                self.stage_out_channels[3],
                self.stage_out_channels[3] // 2,
                kernel_size=3,
                stride=1,
            )
            for _ in range(self.stage_repeats[2] - 1)
        ]

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = x.sequential(self.stage1).pad2d((1, 1, 1, 1)).max_pool2d(3, 2)
        x2 = x.sequential(self.stage2)
        x3 = x2.sequential(self.stage3)
        x4 = x3.sequential(self.stage4)
        return x2, x3, x4


def get_child(parent, key):
    obj = parent
    for k in key.split("."):
        if k.isnumeric():
            obj = obj[int(k)]
        elif isinstance(obj, dict):
            obj = obj[k]
        else:
            obj = getattr(obj, k)
    return obj


if __name__ == "__main__":
    from tinygrad.nn.state import torch_load, safe_save, get_state_dict, get_parameters
    from tinygrad.helpers import dtypes

    net = ShuffleNetV2()

    state_dict = torch_load("./cache/ShuffleNetV2.0.5x.pth.tar")["state_dict"]
    # modify state_dict to match our model
    for key in list(state_dict.keys()):
        if "num_batches_tracked" in key:
            state_dict[key] = Tensor([state_dict[key].numpy().item()])
    for key in list(state_dict.keys()):
        if "first_conv" in key:
            state_dict[key.replace("first_conv", "stage1")] = state_dict[key]
            del state_dict[key]
    for key in list(state_dict.keys()):
        if "branch_main" in key:
            index = int(key.split(".")[4])
            if index == 0:
                state_dict[key.replace("branch_main.0", "cv1")] = state_dict[key]
                del state_dict[key]
            elif index == 1:
                state_dict[key.replace("branch_main.1", "bn1")] = state_dict[key]
                del state_dict[key]
            elif index == 3:
                state_dict[key.replace("branch_main.3", "cv2")] = state_dict[key]
                del state_dict[key]
            elif index == 4:
                state_dict[key.replace("branch_main.4", "bn2")] = state_dict[key]
                del state_dict[key]
            elif index == 5:
                state_dict[key.replace("branch_main.5", "cv3")] = state_dict[key]
                del state_dict[key]
            elif index == 6:
                state_dict[key.replace("branch_main.6", "bn3")] = state_dict[key]
                del state_dict[key]
        if "branch_proj" in key:
            index = int(key.split(".")[4])
            if index == 0:
                state_dict[key.replace("branch_proj.0", "cv4")] = state_dict[key]
                del state_dict[key]
            elif index == 1:
                state_dict[key.replace("branch_proj.1", "bn4")] = state_dict[key]
                del state_dict[key]
            elif index == 2:
                state_dict[key.replace("branch_proj.2", "cv5")] = state_dict[key]
                del state_dict[key]
            elif index == 3:
                state_dict[key.replace("branch_proj.3", "bn5")] = state_dict[key]
                del state_dict[key]
    for key in list(state_dict.keys()):
        if "features" in key:
            index = int(key.split(".")[2])
            if index in range(0, 4):
                state_dict[key.replace(f"features.{index}", f"stage2.{index}")] = state_dict[key]
                del state_dict[key]
            elif index in range(4, 12):
                state_dict[key.replace(f"features.{index}", f"stage3.{index - 4}")] = state_dict[key]
                del state_dict[key]
            elif index in range(12, 16):
                state_dict[key.replace(f"features.{index}", f"stage4.{index - 12}")] = state_dict[key]
                del state_dict[key]

    for key in list(state_dict.keys()):
        if "conv_last" in key:
            continue
        if "classifier" in key:
            continue
        print(key)
        get_child(net, key.replace("module.", "")).assign(
            state_dict[key].to(Device.DEFAULT)
        ).realize()

    for param in get_parameters(state_dict):
        param.assign(param.cast(dtypes.float32)).realize()

    # save state_dict
    safe_save(get_state_dict(net), "./weights/shufflenetv2.safetensors")
