from typing import Tuple

from tinygrad.nn.state import get_parameters, load_state_dict, safe_load
from tinygrad.nn import Conv2d, Embedding, Linear, BatchNorm2d
import onnx
from onnx import numpy_helper, TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.checker import check_model
from onnxconverter_common import float16
import numpy as np

from main import get_foundation, BASE_PATH
from model import Model, Head, ConvBlock, SqueezeExcite, ConvEmbedding
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

def make_BatchNorm2d(n: BatchNorm2d, name: str, x: str):
  weight = numpy_helper.from_array(n.weight.numpy(), name + ".weight")
  bias = numpy_helper.from_array(n.bias.numpy(), name + ".bias")
  mean = numpy_helper.from_array(n.running_mean.numpy(), name + ".mean")
  var = numpy_helper.from_array(n.running_var.numpy(), name + ".var")
  bn = make_node("BatchNormalization", [x, weight.name, bias.name, mean.name, var.name], [name], name=name, epsilon=n.eps)
  return bn, [bn], [weight, bias, mean, var]

def make_ConvBlock(n: ConvBlock, name: str, x: str):
  cv, cv_nodes, cv_weights = make_Conv2d(n.cv, name + ".cv", x)
  if n.activation:
    activation = make_node("LeakyRelu", [cv.output[0]], [name + ".leakyrelu"], name=name + ".leakyrelu", alpha=0.01)
    return activation, [*cv_nodes, activation], [*cv_weights]
  else:
    return cv, [*cv_nodes], [*cv_weights]

def make_SqueezeExcite(n: SqueezeExcite, name: str, x: str):
  avg_pool = make_node("AveragePool", [x], [name + ".avg_pool"], name=name + ".avg_pool", kernel_shape=n.shape, strides=[1, 1])
  s, s_nodes, s_weights = make_Conv2d(n.s, name + ".s", avg_pool.output[0])
  relu = make_node("Relu", [s.output[0]], [name + ".relu"], name=name + ".relu")
  e, e_nodes, e_weights = make_Conv2d(n.e, name + ".e", relu.output[0])
  sigmoid = make_node("Sigmoid", [e.output[0]], [name + ".sigmoid"], name=name + ".sigmoid")
  mul = make_node("Mul", [x, sigmoid.output[0]], [name + ".mul"], name=name + ".mul")
  return mul, [avg_pool, *s_nodes, relu, *e_nodes, sigmoid, mul], [*s_weights, *e_weights]

def make_ConvEmbedding(n: ConvEmbedding, name: str, x: str):
  cv1, cv1_nodes, cv1_weights = make_ConvBlock(n.cv1, name + ".cv1", x)
  se, se_nodes, se_weights = make_SqueezeExcite(n.se, name + ".se", cv1.output[0])
  cv2, cv2_nodes, cv2_weights = make_ConvBlock(n.cv2, name + ".cv2", se.output[0])
  return cv2, [*cv1_nodes, *se_nodes, *cv2_nodes], [*cv1_weights, *se_weights, *cv2_weights]

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
  pads = numpy_helper.from_array(np.array([0, 1, 0, 1], dtype=np.int64), name + ".pads")
  pad = make_node("Pad", [x, pads.name], [name + ".pad"], name=name + ".pad")
  cv, cv_nodes, cv_weights = make_ConvBlock(head.cv, name + ".cv", pad.output[0])

  conv_emb, conv_emb_nodes, conv_emb_weights = make_ConvEmbedding(head.conv_emb, name + ".conv_emb", cv.output[0])
  shape = numpy_helper.from_array(np.array([1, head.joint.weight.shape[1]], dtype=np.int64), name + ".shape")
  reshape = make_node("Reshape", [conv_emb.output[0], shape.name], [name + ".reshape"], name=name + ".reshape")

  color_emb, color_emb_nodes, color_emb_weights = make_Embedding(head.color_emb, name + ".color_emb", color)
  squeeze_axis = numpy_helper.from_array(np.array([1], dtype=np.int64), name + ".squeeze_axis")
  color_emb_squeeze = make_node("Squeeze", [color_emb.output[0], squeeze_axis.name], [name + ".color_emb_squeeze"], name=name + ".color_emb_squeeze")
  add = make_node("Add", [reshape.output[0], color_emb_squeeze.output[0]], [name + ".add"], name=name + ".add")
  joint, joint_nodes, joint_weights = make_Linear(head.joint, name + ".joint", add.output[0])
  joint_lr = make_node("LeakyRelu", [joint.output[0]], [name + ".joint_lr"], name=name + ".joint_lr", alpha=0.01)
  fc1, fc1_nodes, fc1_weights = make_Linear(head.fc1, name + ".fc1", joint_lr.output[0])
  fc1_lr = make_node("LeakyRelu", [fc1.output[0]], [name + ".fc1_lr"], name=name + ".fc1_lr", alpha=0.01)
  l_out, l_out_nodes, l_out_weights = make_Linear(head.l_out, name + ".l_out", fc1_lr.output[0])

  return l_out, [pad, *cv_nodes, *conv_emb_nodes, reshape, *color_emb_nodes, color_emb_squeeze, add, *joint_nodes, joint_lr, *fc1_nodes, fc1_lr, *l_out_nodes], [pads, *cv_weights, *conv_emb_weights, shape, *color_emb_weights, squeeze_axis, *joint_weights, *fc1_weights, *l_out_weights]

def make_Model(model: Model, name: str, x: Tuple[str, str, str], color: str):
  x2, x3, x5 = x
  scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name + ".scales")
  x5_upsample = make_node("Resize", [x5, scales.name], [name + ".x5_upsample"], name=name + ".x5_upsample", mode="nearest")
  x2_avg_pool = make_node("AveragePool", [x2], [name + ".x2_avg_pool"], name=name + ".x2_avg_pool", kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
  concat = make_node("Concat", [x2_avg_pool.output[0], x3, x5_upsample.output[0]], [name + ".concat"], name=name + ".concat", axis=1)
  head, head_nodes, head_weights = make_Head(model.head, name + ".head", concat.output[0], color)
  return head, [x5_upsample, x2_avg_pool, concat, *head_nodes], [scales, *head_weights]

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

  return (b2_2, b3_1, b5_0), [*b1_0_nodes, *b1_1_nodes, *b2_0_nodes, *b2_1_nodes, *b2_2_nodes, *b3_0_nodes, *b3_1_nodes, *b4_0_nodes, *b4_1_nodes, *b5_0_nodes], [*b1_0_weights, *b1_1_weights, *b2_0_weights, *b2_1_weights, *b2_2_weights, *b3_0_weights, *b3_1_weights, *b4_0_weights, *b4_1_weights, *b5_0_weights]

def make_preprocess(name: str, x: str):
  div_const = numpy_helper.from_array(np.array([255], dtype=np.float32), name + ".div_const")
  div = make_node("Div", [x, div_const.name], [name + ".div"], name=name + ".div")
  permute = make_node("Transpose", [div.output[0]], [name + ".permute"], name=name + ".permute", perm=[0, 3, 1, 2])
  return permute, [div, permute], [div_const]

if __name__ == "__main__":
  foundation = get_foundation()
  model = Model()
  # load_state_dict(model, safe_load(str(BASE_PATH / "model.safetensors")))

  print(f"there are {sum(param.numel() for param in get_parameters(model)) / 1e6}M params")

  x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 480, 640, 3])
  color = make_tensor_value_info("color", TensorProto.INT32, [1, 1])
  out = make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])

  preprocess_node, preprocess_nodes, preprocess_weights = make_preprocess("preprocess", x.name)
  foundation_node, foundation_nodes, foundation_weights = make_Darknet(foundation.net, "foundation", preprocess_node.output[0])
  model_node, model_nodes, model_weights = make_Model(model, "model", (foundation_node[0].output[0], foundation_node[1].output[0], foundation_node[2].output[0]), color.name)
  model_node.output[0] = out.name

  graph = make_graph([*preprocess_nodes, *foundation_nodes, *model_nodes], "model", [x, color], [out], [*preprocess_weights, *foundation_weights, *model_weights])
  model = make_model(graph)
  check_model(model)
  onnx.save(model, "model.onnx")

  model_fp16 = float16.convert_float_to_float16(model)
  onnx.save(model_fp16, "model_fp16.onnx")
