from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.nn import Conv2d, Embedding, Linear
from onnx import numpy_helper, TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.checker import check_model
import numpy as np

from main import get_foundation, BASE_PATH
from model import Head, ConvBlock, ConvEmbedding


def make_Conv2d(n: Conv2d, name: str, x: str):
    weight = numpy_helper.from_array(n.weight.numpy(), name + ".weight")
    if n.bias is not None:
        bias = numpy_helper.from_array(n.bias.numpy(), name + ".bias")
        conv = make_node("Conv", [x, weight.name, bias.name], [name], name=name, pads=[n.padding]*4, kernel_shape=n.kernel_size)
        return conv, [conv], [weight, bias]
    else:
        conv = make_node("Conv", [x, weight.name], [name], name=name, pads=[n.padding]*4, kernel_shape=n.kernel_size)
        return conv, [conv], [weight]


def make_ConvBlock(n: ConvBlock, name: str, x: str):
    c1, c1_nodes, c1_weights = make_Conv2d(n.c1, name + ".c1", x)
    mp = make_node("MaxPool", [c1.output[0]], [name + ".mp"], name=name + ".mp", kernel_shape=[2, 2])
    gelu1 = make_node("Mish", [mp.output[0]], [name + ".mish1"], name=name + ".mish1")

    c_res, c_res_nodes, c_res_weights = make_Conv2d(n.c_res, name + ".c_res", gelu1.output[0])

    c2, c2_nodes, c2_weights = make_Conv2d(n.c2, name + ".c2", gelu1.output[0])
    gelu2 = make_node("Mish", [c2.output[0]], [name + ".mish2"], name=name + ".mish2")

    reduce = make_node("Add", [c_res.output[0], gelu2.output[0]], [name + ".reduce"], name=name + ".reduce")
    return reduce, [*c1_nodes, mp, gelu1, *c_res_nodes, *c2_nodes, gelu2, reduce], [*c1_weights, *c_res_weights, *c2_weights]


def make_ConvEmbedding(n: ConvEmbedding, name: str, x: str):
    pre_conv, pre_conv_nodes, pre_conv_weights = make_ConvBlock(n.pre_conv, name + ".pre_conv", x)

    post_conv, post_conv_nodes, post_conv_weights = make_Conv2d(n.post_conv, name + ".post_conv", pre_conv.output[0])

    axes = numpy_helper.from_array(np.array([2, 3]), name + ".axes")
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
    weight = numpy_helper.from_array(n.weight.numpy(), name + ".weight")
    if n.bias is not None:
        bias = numpy_helper.from_array(n.bias.numpy(), name + ".bias")
        matmul = make_node("MatMul", [x, weight.name], [name + ".matmul"], name=name + ".matmul")
        add = make_node("Add", [matmul.output[0], bias.name], [name], name=name)
        return add, [matmul, add], [weight, bias]
    else:
        matmul = make_node("MatMul", [x, weight.name], [name], name=name)
        return matmul, [matmul], [weight]


def make_Head(head: Head, name: str, x: str, color: str):
    conv_emb, conv_emb_nodes, conv_emb_weights = make_ConvEmbedding(head.conv_emb, name + ".conv_emb", x)
    color_emb, color_emb_nodes, color_emb_weights = make_Embedding(head.color_emb, name + ".color_emb", color)
    cat = make_node("Concat", [conv_emb.output[0], color_emb.output[0]], [name + ".cat"], name=name + ".cat", axis=1)
    joint, joint_nodes, joint_weights = make_Linear(head.joint, name + ".joint", cat.output[0])
    l_out_node, l_out_nodes, l_out_weights = make_Linear(head.l_out, name + ".l_out", joint.output[0])

    return l_out_node, [*conv_emb_nodes, *color_emb_nodes, cat, *joint_nodes, *l_out_nodes], [*conv_emb_weights, *color_emb_weights, *joint_weights, *l_out_weights]


if __name__ == "__main__":
    head = Head()
    load_state_dict(head, safe_load(str(BASE_PATH / "model.safetensors")))

    x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 256, 15, 20])
    color = make_tensor_value_info("color", TensorProto.FLOAT, [1, 1])
    out = make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])

    l_out_node, nodes, weights = make_Head(head, "head", "x", "color")
    l_out_node.output[0] = "out"

    graph = make_graph(nodes, "head", [x, color], [out], weights)
    model = make_model(graph)
    check_model(model)

    with open("head.onnx", "wb") as f:
        f.write(model.SerializeToString())
