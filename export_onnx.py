from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.nn import Conv2d, Embedding, Linear, BatchNorm2d
from onnx import numpy_helper, TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.checker import check_model
import numpy as np

from main import get_foundation, BASE_PATH
from model import Head, ConvBlock, ConvEmbedding
from yolov8 import Darknet, Conv_Block as DarknetConvBlock, Bottleneck as DarknetBottleneck, C2f as DarknetC2f, SPPF as DarknetSPPF


def make_Conv2d(n: Conv2d, name: str, x: str):
    weight = numpy_helper.from_array(n.weight.numpy(), name + ".weight")
    if n.bias is not None:
        bias = numpy_helper.from_array(n.bias.numpy(), name + ".bias")
        conv = make_node("Conv", [x, weight.name, bias.name], [name], name=name, pads=[n.padding]*4, kernel_shape=n.kernel_size, group=n.groups, strides=[n.stride]*2)
        return conv, [conv], [weight, bias]
    else:
        conv = make_node("Conv", [x, weight.name], [name], name=name, pads=[n.padding]*4 if isinstance(n.padding, int) else [p for p in n.padding for _ in range(2)][::-1], kernel_shape=n.kernel_size, group=n.groups, strides=[n.stride]*2)
        return conv, [conv], [weight]


def make_ConvBlock(n: ConvBlock, name: str, x: str):
    c1, c1_nodes, c1_weights = make_Conv2d(n.c1, name + ".c1", x)
    mp = make_node("MaxPool", [c1.output[0]], [name + ".mp"], name=name + ".mp", kernel_shape=[2, 2], strides=[2, 2])
    mish1 = make_node("Mish", [mp.output[0]], [name + ".mish1"], name=name + ".mish1")

    c_res, c_res_nodes, c_res_weights = make_Conv2d(n.c_res, name + ".c_res", mish1.output[0])

    c2, c2_nodes, c2_weights = make_Conv2d(n.c2, name + ".c2", mish1.output[0])
    mish2 = make_node("Mish", [c2.output[0]], [name + ".mish2"], name=name + ".mish2")

    reduce = make_node("Add", [c_res.output[0], mish2.output[0]], [name + ".reduce"], name=name + ".reduce")
    return reduce, [*c1_nodes, mp, mish1, *c_res_nodes, *c2_nodes, mish2, reduce], [*c1_weights, *c_res_weights, *c2_weights]


def make_ConvEmbedding(n: ConvEmbedding, name: str, x: str):
    pre_conv, pre_conv_nodes, pre_conv_weights = make_ConvBlock(n.pre_conv, name + ".pre_conv", x)

    post_conv, post_conv_nodes, post_conv_weights = make_Conv2d(n.post_conv, name + ".post_conv", pre_conv.output[0])

    axes = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name + ".axes")
    if n.reduction == "max":
        reduce = make_node("ReduceMax", [post_conv.output[0], axes.name], [name + ".reduce"], name=name + ".reduce", keepdims=0)
    elif n.reduction == "mean":
        reduce = make_node("ReduceMean", [post_conv.output[0], axes.name], [name + ".reduce"], name=name + ".reduce", keepdims=0)
    elif n.reduction == "sum":
        reduce = make_node("ReduceSum", [post_conv.output[0], axes.name], [name + ".reduce"], name=name + ".reduce", keepdims=0)

    return reduce, [*pre_conv_nodes, *post_conv_nodes, reduce], [*pre_conv_weights, *post_conv_weights, axes]


def make_Embedding(n: Embedding, name: str, x: str):
    weight = numpy_helper.from_array(n.weight.numpy(), name + ".weight")
    gather = make_node("Gather", [weight.name, x], [name], name=name)
    return gather, [gather], [weight]


def make_Linear(n: Linear, name: str, x: str):
    weight = numpy_helper.from_array(n.weight.numpy().T, name + ".weight")
    if n.bias is not None:
        bias = numpy_helper.from_array(n.bias.numpy(), name + ".bias")
        matmul = make_node("MatMul", [x, weight.name], [name + ".matmul"], name=name + ".matmul")
        add = make_node("Add", [matmul.output[0], bias.name], [name + ".reduce"], name=name + ".reduce")
        return add, [matmul, add], [weight, bias]
    else:
        matmul = make_node("MatMul", [x, weight.name], [name], name=name)
        return matmul, [matmul], [weight]


def make_Head(head: Head, name: str, x: str, color: str):
    conv_emb, conv_emb_nodes, conv_emb_weights = make_ConvEmbedding(head.conv_emb, name + ".conv_emb", x)
    color_emb, color_emb_nodes, color_emb_weights = make_Embedding(head.color_emb, name + ".color_emb", color)
    color_emb_squeeze_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name + ".color_emb_squeeze_axes")
    color_emb_squeeze = make_node("Squeeze", [color_emb.output[0], color_emb_squeeze_axes.name], [name + ".color_emb_squeeze"], name=name + ".color_emb_squeeze")
    cat = make_node("Concat", [conv_emb.output[0], color_emb_squeeze.output[0]], [name + ".cat"], name=name + ".cat", axis=1)
    joint, joint_nodes, joint_weights = make_Linear(head.joint, name + ".joint", cat.output[0])
    leakyrelu = make_node("LeakyRelu", [joint.output[0]], [name + ".leakyrelu"], name=name + ".leakyrelu", alpha=0.01)
    l_out_node, l_out_nodes, l_out_weights = make_Linear(head.l_out, name + ".l_out", leakyrelu.output[0])

    return l_out_node, [*conv_emb_nodes, *color_emb_nodes, color_emb_squeeze, cat, *joint_nodes, leakyrelu, *l_out_nodes], [*conv_emb_weights, *color_emb_weights, color_emb_squeeze_axes, *joint_weights, *l_out_weights]


def make_BatchNorm2d(n: BatchNorm2d, name: str, x: str):
    weight = numpy_helper.from_array(n.weight.numpy(), name + ".weight")
    bias = numpy_helper.from_array(n.bias.numpy(), name + ".bias")
    mean = numpy_helper.from_array(n.running_mean.numpy(), name + ".mean")
    var = numpy_helper.from_array(n.running_var.numpy(), name + ".var")
    bn = make_node("BatchNormalization", [x, weight.name, bias.name, mean.name, var.name], [name], name=name, epsilon=n.eps)
    return bn, [bn], [weight, bias, mean, var]


def make_DarknetConvBlock(n: DarknetConvBlock, name: str, x: str):
    conv, conv_nodes, conv_weights = make_Conv2d(n.conv, name + ".conv", x)
    bn, bn_nodes, bn_weights = make_BatchNorm2d(n.bn, name + ".bn", conv.output[0])
    sigmoid = make_node("Sigmoid", [bn.output[0]], [name + ".sigmoid"], name=name + ".sigmoid")
    swish = make_node("Mul", [bn.output[0], sigmoid.output[0]], [name + ".swish"], name=name + ".swish")
    return swish, [*conv_nodes, *bn_nodes, sigmoid, swish], [*conv_weights, *bn_weights]


def make_DarknetBottleneck(n: DarknetBottleneck, name: str, x: str):
    cv1, cv1_nodes, cv1_weights = make_DarknetConvBlock(n.cv1, name + ".cv1", x)
    cv2, cv2_nodes, cv2_weights = make_DarknetConvBlock(n.cv2, name + ".cv2", cv1.output[0])
    if n.residual:
        add = make_node("Add", [x, cv2.output[0]], [name + ".add"], name=name + ".add")
        return add, [*cv1_nodes, *cv2_nodes, add], [*cv1_weights, *cv2_weights]
    else:
        return cv2, [*cv1_nodes, *cv2_nodes], [*cv1_weights, *cv2_weights]


def make_DarknetC2f(n: DarknetC2f, name: str, x: str):
    cv1, cv1_nodes, cv1_weights = make_DarknetConvBlock(n.cv1, name + ".cv1", x)
    chunk = make_node("Split", [cv1.output[0]], [name + ".chunk1", name + ".chunk2"], name=name + ".chunk", axis=1, num_outputs=2)

    bottlenecks, bottleneck_nodes, bottleneck_weights, bottleneck_input = [], [], [], chunk.output[-1]
    for i, bottleneck in enumerate(n.bottleneck):
        bottleneck, bottleneck_nodes_, bottleneck_weights_ = make_DarknetBottleneck(bottleneck, name + f".bottleneck{i}", bottleneck_input)
        bottlenecks.append(bottleneck)
        bottleneck_nodes.extend(bottleneck_nodes_)
        bottleneck_weights.extend(bottleneck_weights_)
        bottleneck_input = bottleneck.output[0]

    concat = make_node("Concat", [*chunk.output, *[b.output[0] for b in bottlenecks]], [name + ".concat"], name=name + ".concat", axis=1)

    cv2, cv2_nodes, cv2_weights = make_DarknetConvBlock(n.cv2, name + ".cv2", concat.output[0])
    return cv2, [*cv1_nodes, chunk, *bottleneck_nodes, concat, *cv2_nodes], [*cv1_weights, *bottleneck_weights, *cv2_weights]


def make_DarknetSPPF(n: DarknetSPPF, name: str, x: str):
    cv1, cv1_nodes, cv1_weights = make_DarknetConvBlock(n.cv1, name + ".cv1", x)

    mp1 = make_node("MaxPool", [cv1.output[0]], [name + ".mp1"], name=name + ".mp1", kernel_shape=[n.k, n.k], strides=[1, 1], pads=[n.k // 2]*4)
    mp2 = make_node("MaxPool", [mp1.output[0]], [name + ".mp2"], name=name + ".mp2", kernel_shape=[n.k, n.k], strides=[1, 1], pads=[n.k // 2]*4)
    mp3 = make_node("MaxPool", [mp2.output[0]], [name + ".mp3"], name=name + ".mp3", kernel_shape=[n.k, n.k], strides=[1, 1], pads=[n.k // 2]*4)

    concat = make_node("Concat", [cv1.output[0], mp1.output[0], mp2.output[0], mp3.output[0]], [name + ".concat"], name=name + ".concat", axis=1)

    cv2, cv2_nodes, cv2_weights = make_DarknetConvBlock(n.cv2, name + ".cv2", concat.output[0])
    return cv2, [*cv1_nodes, mp1, mp2, mp3, concat, *cv2_nodes], [*cv1_weights, *cv2_weights]


def make_Darknet(net: Darknet, name: str, x: str):
    b1_0, b1_0_nodes, b1_0_weights = make_DarknetConvBlock(net.b1[0], name + ".b1_0", x)
    b1_1, b1_1_nodes, b1_1_weights = make_DarknetConvBlock(net.b1[1], name + ".b1_1", b1_0.output[0])

    b2_0, b2_0_nodes, b2_0_weights = make_DarknetC2f(net.b2[0], name + ".b2_0", b1_1.output[0])
    b2_1, b2_1_nodes, b2_1_weights = make_DarknetConvBlock(net.b2[1], name + ".b2_1", b2_0.output[0])
    b2_2, b2_2_nodes, b2_2_weights = make_DarknetC2f(net.b2[2], name + ".b2_2", b2_1.output[0])

    b3_0, b3_0_nodes, b3_0_weights = make_DarknetConvBlock(net.b3[0], name + ".b3_0", b2_2.output[0])
    b3_1, b3_1_nodes, b3_1_weights = make_DarknetC2f(net.b3[1], name + ".b3_1", b3_0.output[0])

    b4_0, b4_0_nodes, b4_0_weights = make_DarknetConvBlock(net.b4[0], name + ".b4_0", b3_1.output[0])
    b4_1, b4_1_nodes, b4_1_weights = make_DarknetC2f(net.b4[1], name + ".b4_1", b4_0.output[0])

    b5_0, b5_0_nodes, b5_0_weights = make_DarknetSPPF(net.b5[0], name + ".b5_0", b4_1.output[0])

    return b5_0, [*b1_0_nodes, *b1_1_nodes, *b2_0_nodes, *b2_1_nodes, *b2_2_nodes, *b3_0_nodes, *b3_1_nodes, *b4_0_nodes, *b4_1_nodes, *b5_0_nodes], [*b1_0_weights, *b1_1_weights, *b2_0_weights, *b2_1_weights, *b2_2_weights, *b3_0_weights, *b3_1_weights, *b4_0_weights, *b4_1_weights, *b5_0_weights]


def make_preprocess(name: str, x: str):
    div_const = numpy_helper.from_array(np.array([255], dtype=np.float32), name + ".div_const")
    div = make_node("Div", [x, div_const.name], [name + ".div"], name=name + ".div")
    permute = make_node("Transpose", [div.output[0]], [name + ".permute"], name=name + ".permute", perm=[0, 3, 1, 2])
    return permute, [div, permute], [div_const]


if __name__ == "__main__":
    foundation = get_foundation()
    head = Head()
    load_state_dict(head, safe_load(str(BASE_PATH / "model.safetensors")))

    x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 480, 640, 3])
    color = make_tensor_value_info("color", TensorProto.INT32, [1, 1])
    out = make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])

    preprocess_node, preprocess_nodes, preprocess_weights = make_preprocess("preprocess", "x")
    foundation_node, foundation_nodes, foundation_weights = make_Darknet(foundation.net, "foundation", preprocess_node.output[0])
    head_node, head_nodes, head_weights = make_Head(head, "head", foundation_node.output[0], "color")
    head_node.output[0] = "out"

    graph = make_graph([*preprocess_nodes, *foundation_nodes, *head_nodes], "model", [x, color], [out], [*preprocess_weights, *foundation_weights, *head_weights])
    model = make_model(graph)
    check_model(model)

    with open("model.onnx", "wb") as f:
        f.write(model.SerializeToString())
