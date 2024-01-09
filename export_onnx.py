from typing import Tuple

from tinygrad.nn.state import get_parameters, load_state_dict, safe_load, get_state_dict
from tinygrad.nn import Conv2d, Linear, BatchNorm2d
import onnx
from onnx import numpy_helper, shape_inference
from onnx.onnx_pb import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.checker import check_model
from onnx.version_converter import convert_version
import numpy as np

from main import BASE_PATH
from model import Model, ObjHead, PosHead, DFCAttention, EncoderBlock, EncoderDecoder
from shufflenet import ShuffleNetV2, ShuffleV2Block

def make_Conv2d(n: Conv2d, name: str, x: str):
  weight = numpy_helper.from_array(n.weight.numpy(), name + ".weight")
  if n.bias is not None:
    bias = numpy_helper.from_array(n.bias.numpy(), name + ".bias")
    conv = make_node("Conv", [x, weight.name, bias.name], [name], name=name, pads=[n.padding]*4, kernel_shape=n.kernel_size, group=n.groups, strides=[n.stride]*2 if isinstance(n.stride, int) else n.stride)
    return conv, [conv], [weight, bias]
  else:
    conv = make_node("Conv", [x, weight.name], [name], name=name, pads=[n.padding]*4 if isinstance(n.padding, int) else [p for p in n.padding for _ in range(2)][::-1], kernel_shape=n.kernel_size, group=n.groups, strides=[n.stride]*2 if isinstance(n.stride, int) else n.stride) # type: ignore
    return conv, [conv], [weight]

def make_BatchNorm2d(n: BatchNorm2d, name: str, x: str):
  assert n.weight is not None and n.bias is not None
  weight = numpy_helper.from_array(n.weight.numpy(), name + ".weight")
  bias = numpy_helper.from_array(n.bias.numpy(), name + ".bias")
  mean = numpy_helper.from_array(n.running_mean.numpy(), name + ".mean")
  var = numpy_helper.from_array(n.running_var.numpy(), name + ".var")
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

def make_mish(name: str, x: str):
  softplus = make_node("Softplus", [x], [name + ".softplus"], name=name + ".softplus")
  tanh = make_node("Tanh", [softplus.output[0]], [name + ".tanh"], name=name + ".tanh")
  mul = make_node("Mul", [x, tanh.output[0]], [name], name=name)
  return mul, [softplus, tanh, mul], []

def make_channel_shuffle(channels: int, height: int, width: int, name: str, x: str):
  shape1 = numpy_helper.from_array(np.array([1 * channels // 2, 2, height * width], dtype=np.int64), name + ".shape1")
  reshape1 = make_node("Reshape", [x, shape1.name], [name + ".reshape1"], name=name + ".reshape1")
  transpose1 = make_node("Transpose", [reshape1.output[0]], [name + ".transpose1"], name=name + ".transpose1", perm=[1, 0, 2])
  shape2 = numpy_helper.from_array(np.array([2, 1, channels // 2, height, width], dtype=np.int64), name + ".shape2")
  reshape2 = make_node("Reshape", [transpose1.output[0], shape2.name], [name + ".reshape2"], name=name + ".reshape2")
  split = make_node("Split", [reshape2.output[0]], [name + ".split1", name + ".split2"], name=name + ".split", num_outputs=2, axis=0)
  squeeze_axis = numpy_helper.from_array(np.array([0], dtype=np.int64), name=name + ".squeeze_axis")
  squeeze1 = make_node("Squeeze", [split.output[0], squeeze_axis.name], [name + ".squeeze1"], name=name + ".squeeze1")
  squeeze2 = make_node("Squeeze", [split.output[1], squeeze_axis.name], [name + ".squeeze2"], name=name + ".squeeze2")
  return (squeeze1, squeeze2), [reshape1, transpose1, reshape2, split, squeeze1, squeeze2], [shape1, shape2, squeeze_axis]

# TODO: this is kinda hacky
channel_to_hw = {48: 40, 96: 20, 192: 10}
def make_ShuffleV2Block(n: ShuffleV2Block, name: str, x: str):
  if n.stride == 1:
    channel_shuffle, channel_shuffle_nodes, channel_shuffle_weights = make_channel_shuffle(n.outp, channel_to_hw[n.outp], channel_to_hw[n.outp], name + ".channel_shuffle", x)
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
  stage1_norm, stage1_norm_nodes, stage1_norm_weights = make_BatchNorm2d(n.stage1[2], name + ".stage1_norm", stage1_conv.output[0])
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
  stage5_norm, stage5_norm_nodes, stage5_norm_weights = make_BatchNorm2d(n.stage5[2], name + ".stage5_norm", stage5_conv.output[0])
  stage5_relu = make_node("Relu", [stage5_norm.output[0]], [name + ".stage5_relu"], name=name + ".stage5_relu")
  return stage5_relu, [*stage1_conv_nodes, *stage1_norm_nodes, stage1_relu, max_pool, *stage2_nodes, *stage3_nodes, *stage4_nodes, *stage5_conv_nodes, *stage5_norm_nodes, stage5_relu], [*stage1_conv_weights, *stage1_norm_weights, *stage2_weights, *stage3_weights, *stage4_weights, *stage5_conv_weights, *stage5_norm_weights]

def make_DFCAttention(n: DFCAttention, name: str, x: str):
  downsample = make_node("AveragePool", [x], [name + ".downsample"], name=name + ".downsample", kernel_shape=[2, 2], strides=[2, 2])
  cv, cv_nodes, cv_weights = make_Conv2d(n.cv, name + ".cv", downsample.output[0])
  norm, norm_nodes, norm_weights = make_BatchNorm2d(n.norm, name + ".norm", cv.output[0])
  hcv, hcv_nodes, hcv_weights = make_Conv2d(n.hcv, name + ".hcv", norm.output[0])
  hnorm, hnorm_nodes, hnorm_weights = make_BatchNorm2d(n.hnorm, name + ".hnorm", hcv.output[0])
  vcv, vcv_nodes, vcv_weights = make_Conv2d(n.vcv, name + ".vcv", hnorm.output[0])
  vnorm, vnorm_nodes, vnorm_weights = make_BatchNorm2d(n.vnorm, name + ".vnorm", vcv.output[0])
  sigmoid = make_node("Sigmoid", [vnorm.output[0]], [name + ".sigmoid"], name=name + ".sigmoid")
  scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name + ".upsample_scales")
  upsample = make_node("Resize", [sigmoid.output[0], "", scales.name], [name + ".upsample"], name=name + ".upsample", mode="nearest")
  return upsample, [downsample, *cv_nodes, *norm_nodes, *hcv_nodes, *hnorm_nodes, *vcv_nodes, *vnorm_nodes, sigmoid, upsample], [*cv_weights, *norm_weights, *hcv_weights, *hnorm_weights, *vcv_weights, *vnorm_weights, scales]

def make_EncoderBlock(n: EncoderBlock, name: str, x: str):
  attention, attention_nodes, attention_weights = make_DFCAttention(n.attention, name + ".attention", x)
  mul = make_node("Mul", [x, attention.output[0]], [name + ".mul"], name=name + ".mul")
  add1 = make_node("Add", [x, mul.output[0]], [name + ".add1"], name=name + ".add1")
  norm1, norm1_nodes, norm1_weights = make_BatchNorm2d(n.norm1, name + ".norm1", add1.output[0])
  cv1, cv1_nodes, cv1_weights = make_Conv2d(n.cv1, name + ".cv1", norm1.output[0])
  nl1, nl1_nodes, nl1_weights = make_mish(name + ".nl1", cv1.output[0])
  cv2, cv2_nodes, cv2_weights = make_Conv2d(n.cv2, name + ".cv2", nl1.output[0])
  add2 = make_node("Add", [nl1.output[0], cv2.output[0]], [name + ".add2"], name=name + ".add2")
  norm2, norm2_nodes, norm2_weights = make_BatchNorm2d(n.norm2, name + ".norm2", add2.output[0])
  return norm2, [*attention_nodes, mul, add1, *norm1_nodes, *cv1_nodes, *nl1_nodes, *cv2_nodes, add2, *norm2_nodes], [*attention_weights, *norm1_weights, *cv1_weights, *nl1_weights, *cv2_weights, *norm2_weights]

def make_EncoderDecoder(n: EncoderDecoder, name: str, x: str):
  encoders, encoder_nodes, encoder_weights, encoder_input = [], [], [], x
  for i, encoder in enumerate(n.encoders):
    encoder, encoder_nodes_, encoder_weights_ = make_EncoderBlock(encoder, name + f".encoder{i}", encoder_input)
    encoders.append(encoder)
    encoder_nodes.extend(encoder_nodes_)
    encoder_weights.extend(encoder_weights_)
    encoder_input = encoder.output[0]

  cv1, cv1_nodes, cv1_weights = make_Conv2d(n.cv1, name + ".cv1", encoder_input)
  norm1, norm1_nodes, norm1_weights = make_BatchNorm2d(n.norm1, name + ".norm1", cv1.output[0])
  nl1, nl1_nodes, nl1_weights = make_mish(name + ".nl1", norm1.output[0])
  cv2, cv2_nodes, cv2_weights = make_Conv2d(n.cv2, name + ".cv2", nl1.output[0])
  norm2, norm2_nodes, norm2_weights = make_BatchNorm2d(n.norm2, name + ".norm2", cv2.output[0])
  nl2, nl2_nodes, nl2_weights = make_mish(name + ".nl2", norm2.output[0])
  cv_out, cv_out_nodes, cv_out_weights = make_Conv2d(n.cv_out, name + ".cv_out", nl2.output[0])
  avg_pool = make_node("AveragePool", [cv_out.output[0]], [name + ".avg_pool"], name=name + ".avg_pool", kernel_shape=[2, 2], strides=[2, 2])
  output_shape = numpy_helper.from_array(np.array([1, n.dim], dtype=np.int64), name + ".output_shape")
  reshape = make_node("Reshape", [avg_pool.output[0], output_shape.name], [name + ".reshape"], name=name + ".reshape")
  return reshape, [*encoder_nodes, *cv1_nodes, *norm1_nodes, *nl1_nodes, *cv2_nodes, *norm2_nodes, *nl2_nodes, *cv_out_nodes, avg_pool, reshape], [*encoder_weights, *cv1_weights, *norm1_weights, *nl1_weights, *cv2_weights, *norm2_weights, *nl2_weights, *cv_out_weights, output_shape]

def make_ObjHead(n: ObjHead, name: str, x: str):
  l1, l1_nodes, l1_weights = make_Linear(n.l1, name + ".l1", x)
  sigmoid = make_node("Sigmoid", [l1.output[0]], [name + ".sigmoid"], name=name + ".sigmoid")
  output_shape = numpy_helper.from_array(np.array([1, n.l1.weight.shape[0] // 2, 1], dtype=np.int64), name + ".output_shape")
  reshape = make_node("Reshape", [sigmoid.output[0], output_shape.name], [name + ".reshape"], name=name + ".reshape")
  return reshape, [*l1_nodes, sigmoid, reshape], [*l1_weights, output_shape]

def make_PosHead(n: PosHead, name: str, x: str):
  l1, l1_nodes, l1_weights = make_Linear(n.l1, name + ".l1", x)
  nl1, nl1_nodes, nl1_weights = make_mish(name + ".nl1", l1.output[0])
  l2, l2_nodes, l2_weights = make_Linear(n.l2, name + ".l2", nl1.output[0])
  nl2, nl2_nodes, nl2_weights = make_mish(name + ".nl2", l2.output[0])
  l3, l3_nodes, l3_weights = make_Linear(n.l3, name + ".l3", nl2.output[0])
  sigmoid = make_node("Sigmoid", [l3.output[0]], [name + ".sigmoid"], name=name + ".sigmoid")
  output_shape = numpy_helper.from_array(np.array([1, n.l3.weight.shape[0] // 2, 2], dtype=np.int64), name + ".output_shape")
  reshape = make_node("Reshape", [sigmoid.output[0], output_shape.name], [name + ".reshape"], name=name + ".reshape")
  return reshape, [*l1_nodes, *nl1_nodes, *l2_nodes, *nl2_nodes, *l3_nodes, sigmoid, reshape], [*l1_weights, *nl1_weights, *l2_weights, *nl2_weights, *l3_weights, output_shape]

def make_Model(model: Model, name: str, x: str):
  backbone, backbone_nodes, backbone_weights = make_ShuffleNetV2(model.backbone, name + ".backbone", x)
  input_conv, input_conv_nodes, input_conv_weights = make_Conv2d(model.input_conv, name + ".input_conv", backbone.output[0])
  encdec, encdec_nodes, encdec_weights = make_EncoderDecoder(model.encdec, name + ".encdec", input_conv.output[0])
  obj_head, obj_head_nodes, obj_head_weights = make_ObjHead(model.obj_head, name + ".obj_head", encdec.output[0])
  pos_head, pos_head_nodes, pos_head_weights = make_PosHead(model.pos_head, name + ".pos_head", encdec.output[0])
  return (obj_head, pos_head), [*backbone_nodes, *input_conv_nodes, *encdec_nodes, *obj_head_nodes, *pos_head_nodes], [*backbone_weights, *input_conv_weights, *encdec_weights, *obj_head_weights, *pos_head_weights]

def make_preprocess(name: str, x: str):
  div_const = numpy_helper.from_array(np.array([255], dtype=np.float16), name + ".div_const")
  div = make_node("Div", [x, div_const.name], [name + ".div"], name=name + ".div")
  permute = make_node("Transpose", [div.output[0]], [name + ".permute"], name=name + ".permute", perm=[0, 3, 1, 2])
  return permute, [div, permute], [div_const]

if __name__ == "__main__":
  model = Model()
  load_state_dict(model, safe_load(str(BASE_PATH / "model.safetensors")))
  for key, param in get_state_dict(model).items():
    if "bn" in key: continue
    param.assign(param.half()).realize()

  print(f"there are {sum(param.numel() for param in get_parameters(model)) / 1e6}M params") # type: ignore

  x = make_tensor_value_info("x", TensorProto.FLOAT16, [1, 320, 320, 3])
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
  model = convert_version(model, 18)
  check_model(model, True)
  onnx.save(model, "model.onnx")
