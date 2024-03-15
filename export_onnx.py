from typing import Tuple

from tinygrad.nn.state import get_parameters, load_state_dict, safe_load, get_state_dict
from tinygrad.nn import Conv2d, Linear, BatchNorm2d
from tinygrad.helpers import getenv
import onnx
from onnx import numpy_helper, shape_inference
from onnx.onnx_pb import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.checker import check_model
import numpy as np

from main import BASE_PATH
from model import Model, ObjHead, PosHead, Head, Neck, SE, FFN, FFNBlock
from shufflenet import ShuffleNetV2, ShuffleV2Block

def make_Conv2d(n: Conv2d, name: str, x: str):
  weight = numpy_helper.from_array(n.weight.numpy(), name + ".weight")
  if n.bias is not None:
    bias = numpy_helper.from_array(n.bias.numpy(), name + ".bias")
    conv = make_node("Conv", [x, weight.name, bias.name], [name], name=name, pads=[n.padding]*4, kernel_shape=n.kernel_size, group=n.groups, strides=[n.stride]*2 if isinstance(n.stride, int) else n.stride)
    return conv, [conv], [weight, bias]
  else:
    conv = make_node("Conv", [x, weight.name], [name], name=name, pads=[n.padding]*4 if isinstance(n.padding, int) else [p for _ in range(2) for p in n.padding], kernel_shape=n.kernel_size, group=n.groups, strides=[n.stride]*2 if isinstance(n.stride, int) else n.stride) # type: ignore
    return conv, [conv], [weight]

def make_BatchNorm2d(n: BatchNorm2d, name: str, x: str):
  assert n.weight is not None and n.bias is not None
  weight = numpy_helper.from_array(n.weight.numpy(), name + ".weight")
  bias = numpy_helper.from_array(n.bias.numpy(), name + ".bias")
  mean = numpy_helper.from_array(n.running_mean.numpy(), name + ".mean")
  var = numpy_helper.from_array(n.running_var.numpy(), name + ".var")
  if getenv("HALFBN") == 0:
    cast1 = make_node("Cast", [x], [name + ".cast1"], name=name + ".cast1", to=TensorProto.FLOAT)
    bn = make_node("BatchNormalization", [cast1.output[0], weight.name, bias.name, mean.name, var.name], [name], name=name, epsilon=n.eps)
    cast2 = make_node("Cast", [bn.output[0]], [name + ".cast2"], name=name + ".cast2", to=TensorProto.FLOAT16)
    return cast2, [cast1, bn, cast2], [weight, bias, mean, var]
  else:
    bn = make_node("BatchNormalization", [x, weight.name, bias.name, mean.name, var.name], [name], name=name, epsilon=n.eps)
    return bn, [bn], [weight, bias, mean, var]

def make_Linear(n: Linear, name: str, x: str):
  weight = numpy_helper.from_array(n.weight.numpy().T, name + ".weight")
  if n.bias is not None:
    bias = numpy_helper.from_array(n.bias.numpy(), name + ".bias")
    gemm = make_node("Gemm", [x, weight.name, bias.name], [name], name=name)
    return gemm, [gemm], [weight, bias]
  else:
    gemm = make_node("Gemm", [x, weight.name], [name], name=name)
    return gemm, [gemm], [weight]

def make_SE(n: SE, name: str, x: str):
  avgpool = make_node("AveragePool", [x], [name + ".avgpool"], name=name + ".avgpool", kernel_shape=[4, 8], strides=[4, 8])
  cv1, cv1_nodes, cv1_weights = make_Conv2d(n.cv1, name + ".cv1", avgpool.output[0])
  relu = make_node("Relu", [cv1.output[0]], [name + ".relu"], name=name + ".relu")
  cv2, cv2_nodes, cv2_weights = make_Conv2d(n.cv2, name + ".cv2", relu.output[0])
  sigmoid = make_node("HardSigmoid", [cv2.output[0]], [name + ".sigmoid"], name=name + ".sigmoid")
  mul = make_node("Mul", [x, sigmoid.output[0]], [name], name=name)
  return mul, [avgpool, *cv1_nodes, relu, *cv2_nodes, sigmoid, mul], [*cv1_weights, *cv2_weights]

def make_FFNBlock(n: FFNBlock, name: str, x: str):
  l1, l1_nodes, l1_weights = make_Linear(n.l1, name + ".l1", x)
  r1 = make_node("Relu", [l1.output[0]], [name + ".r1"], name=name + ".r1")
  l2, l2_nodes, l2_weights = make_Linear(n.l2, name + ".l2", r1.output[0])
  add = make_node("Add", [x, l2.output[0]], [name], name=name)
  r2 = make_node("Relu", [add.output[0]], [name + ".r2"], name=name + ".r2")
  return r2, [*l1_nodes, r1, *l2_nodes, add, r2], [*l1_weights, *l2_weights]

def make_FFN(n: FFN, name: str, x: str):
  blocks, nodes, weights, input_ = [], [], [], x
  for i, block in enumerate(n.blocks):
    block, block_nodes, block_weights = make_FFNBlock(block, name + f".block{i}", input_)
    blocks.append(block)
    nodes.extend(block_nodes)
    weights.extend(block_weights)
    input_ = block.output[0]
  return blocks[-1], nodes, weights

def make_channel_shuffle(channels: int, height: int, width: int, name: str, x: str):
  shape1 = numpy_helper.from_array(np.array([1 * channels // 2, 2, height * width], dtype=np.int64), name + ".shape1")
  reshape1 = make_node("Reshape", [x, shape1.name], [name + ".reshape1"], name=name + ".reshape1")
  transpose1 = make_node("Transpose", [reshape1.output[0]], [name + ".transpose1"], name=name + ".transpose1", perm=[1, 0, 2])
  shape2 = numpy_helper.from_array(np.array([2, 1, channels // 2, height, width], dtype=np.int64), name + ".shape2")
  reshape2 = make_node("Reshape", [transpose1.output[0], shape2.name], [name + ".reshape2"], name=name + ".reshape2")
  split = make_node("Split", [reshape2.output[0]], [name + ".split1", name + ".split2"], name=name + ".split", axis=0)
  squeeze_axis = numpy_helper.from_array(np.array([0], dtype=np.int64), name=name + ".squeeze_axis")
  squeeze1 = make_node("Squeeze", [split.output[0], squeeze_axis.name], [name + ".squeeze1"], name=name + ".squeeze1")
  squeeze2 = make_node("Squeeze", [split.output[1], squeeze_axis.name], [name + ".squeeze2"], name=name + ".squeeze2")
  return (squeeze1, squeeze2), [reshape1, transpose1, reshape2, split, squeeze1, squeeze2], [shape1, shape2, squeeze_axis]

# TODO: this is kinda hacky
channel_to_hw = {48: (16, 32), 96: (8, 16), 192: (4, 8)}
def make_ShuffleV2Block(n: ShuffleV2Block, name: str, x: str):
  if n.stride == 1:
    channel_shuffle, channel_shuffle_nodes, channel_shuffle_weights = make_channel_shuffle(n.outp, channel_to_hw[n.outp][0], channel_to_hw[n.outp][1], name + ".channel_shuffle", x)
    cv1, cv1_nodes, cv1_weights = make_Conv2d(n.cv1, name + ".cv1", channel_shuffle[1].output[0])
    bn1, bn1_nodes, bn1_weights = make_BatchNorm2d(n.bn1, name + ".bn1", cv1.output[0])
    r1 = make_node("Relu", [bn1.output[0]], [name + ".r1"], name=name + ".r1")
    cv2, cv2_nodes, cv2_weights = make_Conv2d(n.cv2, name + ".cv2", r1.output[0])
    bn2, bn2_nodes, bn2_weights = make_BatchNorm2d(n.bn2, name + ".bn2", cv2.output[0])
    cv3, cv3_nodes, cv3_weights = make_Conv2d(n.cv3, name + ".cv3", bn2.output[0])
    bn3, bn3_nodes, bn3_weights = make_BatchNorm2d(n.bn3, name + ".bn3", cv3.output[0])
    r2 = make_node("Relu", [bn3.output[0]], [name + ".r2"], name=name + ".r2")
    concat = make_node("Concat", [channel_shuffle[0].output[0], r2.output[0]], [name + ".concat"], name=name + ".concat", axis=1)
    return concat, [*channel_shuffle_nodes, *cv1_nodes, *bn1_nodes, r1, *cv2_nodes, *bn2_nodes, *cv3_nodes, *bn3_nodes, r2, concat], [*channel_shuffle_weights, *cv1_weights, *bn1_weights, *cv2_weights, *bn2_weights, *cv3_weights, *bn3_weights]
  elif n.stride == 2:
    cv4, cv4_nodes, cv4_weights = make_Conv2d(n.cv4, name + ".cv4", x)
    bn4, bn4_nodes, bn4_weights = make_BatchNorm2d(n.bn4, name + ".bn4", cv4.output[0])
    cv5, cv5_nodes, cv5_weights = make_Conv2d(n.cv5, name + ".cv5", bn4.output[0])
    bn5, bn5_nodes, bn5_weights = make_BatchNorm2d(n.bn5, name + ".bn5", cv5.output[0])
    r1 = make_node("Relu", [bn5.output[0]], [name + ".r1"], name=name + ".r1")
    cv1, cv1_nodes, cv1_weights = make_Conv2d(n.cv1, name + ".cv1", x)
    bn1, bn1_nodes, bn1_weights = make_BatchNorm2d(n.bn1, name + ".bn1", cv1.output[0])
    r2 = make_node("Relu", [bn1.output[0]], [name + ".r2"], name=name + ".r2")
    cv2, cv2_nodes, cv2_weights = make_Conv2d(n.cv2, name + ".cv2", r2.output[0])
    bn2, bn2_nodes, bn2_weights = make_BatchNorm2d(n.bn2, name + ".bn2", cv2.output[0])
    cv3, cv3_nodes, cv3_weights = make_Conv2d(n.cv3, name + ".cv3", bn2.output[0])
    bn3, bn3_nodes, bn3_weights = make_BatchNorm2d(n.bn3, name + ".bn3", cv3.output[0])
    r3 = make_node("Relu", [bn3.output[0]], [name + ".r3"], name=name + ".r3")
    concat = make_node("Concat", [r1.output[0], r3.output[0]], [name + ".concat"], name=name + ".concat", axis=1)
    return concat, [*cv4_nodes, *bn4_nodes, *cv5_nodes, *bn5_nodes, r1, *cv1_nodes, *bn1_nodes, r2, *cv2_nodes, *bn2_nodes, *cv3_nodes, *bn3_nodes, r3, concat], [*cv4_weights, *bn4_weights, *cv5_weights, *bn5_weights, *cv1_weights, *bn1_weights, *cv2_weights, *bn2_weights, *cv3_weights, *bn3_weights]
  raise Exception("Invalid stride", n.stride)

def make_ShuffleNetV2(n: ShuffleNetV2, name: str, x: str):
  stage1_conv, stage1_conv_nodes, stage1_conv_weights = make_Conv2d(n.stage1[0], name + ".stage1_conv", x)
  stage1_norm, stage1_norm_nodes, stage1_norm_weights = make_BatchNorm2d(n.stage1[1], name + ".stage1_norm", stage1_conv.output[0])
  stage1_relu = make_node("Relu", [stage1_norm.output[0]], [name + ".stage1_relu"], name=name + ".stage1_relu")

  max_pool = make_node("MaxPool", [stage1_relu.output[0]], [name + ".max_pool"], name=name + ".max_pool", kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])

  stage2, stage2_nodes, stage2_weights, stage2_input = [], [], [], max_pool.output[0]
  for i, block in enumerate(n.stage2):
    block, block_nodes, block_weights = make_ShuffleV2Block(block, name + f".stage2_{i}", stage2_input)
    stage2.append(block)
    stage2_nodes.extend(block_nodes)
    stage2_weights.extend(block_weights)
    stage2_input = block.output[0]

  stage3, stage3_nodes, stage3_weights, stage3_input = [], [], [], stage2_input
  for i, block in enumerate(n.stage3):
    block, block_nodes, block_weights = make_ShuffleV2Block(block, name + f".stage3_{i}", stage3_input)
    stage3.append(block)
    stage3_nodes.extend(block_nodes)
    stage3_weights.extend(block_weights)
    stage3_input = block.output[0]

  stage4, stage4_nodes, stage4_weights, stage4_input = [], [], [], stage3_input
  for i, block in enumerate(n.stage4):
    block, block_nodes, block_weights = make_ShuffleV2Block(block, name + f".stage4_{i}", stage4_input)
    stage4.append(block)
    stage4_nodes.extend(block_nodes)
    stage4_weights.extend(block_weights)
    stage4_input = block.output[0]

  stage5_conv, stage5_conv_nodes, stage5_conv_weights = make_Conv2d(n.stage5[0], name + ".stage5_conv", stage4_input)
  stage5_norm, stage5_norm_nodes, stage5_norm_weights = make_BatchNorm2d(n.stage5[1], name + ".stage5_norm", stage5_conv.output[0])
  stage5_relu = make_node("Relu", [stage5_norm.output[0]], [name + ".stage5_relu"], name=name + ".stage5_relu")

  return stage5_relu, [*stage1_conv_nodes, *stage1_norm_nodes, stage1_relu, max_pool, *stage2_nodes, *stage3_nodes, *stage4_nodes, *stage5_conv_nodes, *stage5_norm_nodes, stage5_relu], [*stage1_conv_weights, *stage1_norm_weights, *stage2_weights, *stage3_weights, *stage4_weights, *stage5_conv_weights, *stage5_norm_weights]

def make_Neck(n: Neck, name: str, x: str):
  se, se_nodes, se_weights = make_SE(n.se, name + ".se", x)
  conv, conv_nodes, conv_weights = make_Conv2d(n.conv, name + ".conv", se.output[0])
  bn, bn_nodes, bn_weights = make_BatchNorm2d(n.bn, name + ".bn", conv.output[0])
  flatten = make_node("Flatten", [bn.output[0]], [name + ".flatten"], name=name + ".flatten", axis=1)
  proj, proj_nodes, proj_weights = make_Linear(n.proj, name + ".proj", flatten.output[0])
  nl1 = make_node("Relu", [proj.output[0]], [name + ".nl1"], name=name + ".nl1")
  ffn, ffn_nodes, ffn_weights = make_FFN(n.ffn, name + ".ffn", nl1.output[0])
  return ffn, [*se_nodes, *conv_nodes, *bn_nodes, flatten, *proj_nodes, nl1, *ffn_nodes], [*se_weights, *conv_weights, *bn_weights, *proj_weights, *ffn_weights]

def make_ObjHead(n: ObjHead, name: str, x: str):
  proj, proj_nodes, proj_weights = make_Linear(n.proj, name + ".proj", x)
  proj_nl = make_node("Relu", [proj.output[0]], [name + ".proj_nl"], name=name + ".proj_nl")
  l1, l1_nodes, l1_weights = make_Linear(n.l1, name + ".l1", proj_nl.output[0])
  nl1 = make_node("Relu", [l1.output[0]], [name + ".nl1"], name=name + ".nl1")
  l2, l2_nodes, l2_weights = make_Linear(n.l2, name + ".l2", nl1.output[0])
  add = make_node("Add", [proj_nl.output[0], l2.output[0]], [name + ".add"], name=name + ".add")
  nl2 = make_node("Relu", [add.output[0]], [name + ".nl2"], name=name + ".nl2")
  out, out_nodes, out_weights = make_Linear(n.out, name + ".out", nl2.output[0])
  sigmoid = make_node("Sigmoid", [out.output[0]], [name + ".sigmoid"], name=name + ".sigmoid")
  output_shape = numpy_helper.from_array(np.array([1, n.out.weight.shape[0] // 2, 1], dtype=np.int64), name + ".output_shape")
  reshape = make_node("Reshape", [sigmoid.output[0], output_shape.name], [name + ".reshape"], name=name + ".reshape")
  return reshape, [*proj_nodes, proj_nl, *l1_nodes, nl1, *l2_nodes, add, nl2, *out_nodes, sigmoid, reshape], [*proj_weights, *l1_weights, *l2_weights, *out_weights, output_shape]

def make_PosHead(n: PosHead, name: str, x: str):
  proj, proj_nodes, proj_weights = make_Linear(n.proj, name + ".proj", x)
  proj_nl = make_node("Relu", [proj.output[0]], [name + ".proj_nl"], name=name + ".proj_nl")
  l1, l1_nodes, l1_weights = make_Linear(n.l1, name + ".l1", proj_nl.output[0])
  nl1 = make_node("Relu", [l1.output[0]], [name + ".nl1"], name=name + ".nl1")
  l2, l2_nodes, l2_weights = make_Linear(n.l2, name + ".l2", nl1.output[0])
  add = make_node("Add", [proj_nl.output[0], l2.output[0]], [name + ".add"], name=name + ".add")
  nl2 = make_node("Relu", [add.output[0]], [name + ".nl2"], name=name + ".nl2")
  out, out_nodes, out_weights = make_Linear(n.out, name + ".out", nl2.output[0])
  sigmoid = make_node("Sigmoid", [out.output[0]], [name + ".sigmoid"], name=name + ".sigmoid")
  output_shape = numpy_helper.from_array(np.array([1, n.out.weight.shape[0] // 2, 2], dtype=np.int64), name + ".output_shape")
  reshape = make_node("Reshape", [sigmoid.output[0], output_shape.name], [name + ".reshape"], name=name + ".reshape")
  return reshape, [*proj_nodes, proj_nl, *l1_nodes, nl1, *l2_nodes, add, nl2, *out_nodes, sigmoid, reshape], [*proj_weights, *l1_weights, *l2_weights, *out_weights, output_shape]

def make_Head(n: Head, name: str, x: str):
  obj_head, obj_head_nodes, obj_head_weights = make_ObjHead(n.obj, name + ".obj", x)
  pos_head, pos_head_nodes, pos_head_weights = make_PosHead(n.pos, name + ".pos", x)
  return (obj_head, pos_head), [*obj_head_nodes, *pos_head_nodes], [*obj_head_weights, *pos_head_weights]

def make_Model(model: Model, name: str, x: str):
  backbone, backbone_nodes, backbone_weights = make_ShuffleNetV2(model.backbone, name + ".backbone", x)
  neck, neck_nodes, neck_weights = make_Neck(model.neck, name + ".neck", backbone.output[0])
  head, head_nodes, head_weights = make_Head(model.head, name + ".head", neck.output[0])

  return head, [*backbone_nodes, *neck_nodes, *head_nodes], [*backbone_weights, *neck_weights, *head_weights]

def make_preprocess(name: str, x: str):
  div_const = numpy_helper.from_array(np.array([255], dtype=np.float16), name + ".div_const")
  div = make_node("Div", [x, div_const.name], [name + ".div"], name=name + ".div")
  permute = make_node("Transpose", [div.output[0]], [name + ".permute"], name=name + ".permute", perm=[0, 3, 1, 2])
  return permute, [div, permute], [div_const]

if __name__ == "__main__":
  model = Model()
  load_state_dict(model, safe_load(str(BASE_PATH / "model.safetensors")))
  for key, param in get_state_dict(model).items():
    if getenv("HALFBN") == 0:
      if "norm" in key: continue
      if "bn" in key: continue
      if "stage1.1" in key: continue
      if "stage5.1" in key: continue
    param.assign(param.half()).realize()

  print(f"there are {sum(param.numel() for param in get_parameters(model)) / 1e6}M params") # type: ignore
  print(f"{sum(param.numel() for param in get_parameters(model.backbone)) / 1e6}M params are from the backbone") # type: ignore

  x = make_tensor_value_info("x", TensorProto.FLOAT16, [1, 128, 256, 3])
  x_obj = make_tensor_value_info("x_obj", TensorProto.FLOAT16, [1, 1, 1])
  x_pos = make_tensor_value_info("x_pos", TensorProto.FLOAT16, [1, 1, 2])

  preprocess_node, preprocess_nodes, preprocess_weights = make_preprocess("preprocess", x.name)
  model_node, model_nodes, model_weights = make_Model(model, "model", preprocess_node.output[0])
  model_node[0].output[0] = x_obj.name
  model_node[1].output[0] = x_pos.name

  graph = make_graph([*preprocess_nodes, *model_nodes], "model", [x], [x_obj, x_pos], [*preprocess_weights, *model_weights])
  model = make_model(graph)
  del model.opset_import[:]
  opset = model.opset_import.add()
  opset.domain = ""
  opset.version = 14
  model = shape_inference.infer_shapes(model)
  check_model(model, True)
  onnx.save(model, "model.onnx")
