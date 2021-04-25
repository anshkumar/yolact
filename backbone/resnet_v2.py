"""ResNet v2 models for Keras.
Reference:
  - [Identity Mappings in Deep Residual Networks]
    (https://arxiv.org/abs/1603.05027) (CVPR 2016)
https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/applications/resnet_v2.py
"""
from backbone import resnet

def ResNet50V2(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    dcn_layers=[False, False, False, False],
    classifier_activation='softmax'):
  """Instantiates the ResNet50V2 architecture."""
  if not isinstance(dcn_layers, list):
    raise ValueError("dcn_layers must be a list of booleans.")

  if True in dcn_layers:
    if not file_io.file_exists_v2(weights):
      weights = None

  def stack_fn(x):
    x = resnet.stack2(x, 64, 3, use_dcn=dcn_layers[0], name='conv2')
    x = resnet.stack2(x, 128, 4, use_dcn=dcn_layers[1], name='conv3')
    x = resnet.stack2(x, 256, 6, use_dcn=dcn_layers[2], name='conv4')
    return resnet.stack2(x, 512, 3, stride1=1, use_dcn=dcn_layers[3], 
    					 name='conv5')

  return resnet.ResNet(
      stack_fn,
      True,
      True,
      'resnet50v2',
      include_top,
      weights,
      input_tensor,
      input_shape,
      pooling,
      classes,
      classifier_activation=classifier_activation)

def ResNet101V2(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    dcn_layers=[False, False, False, False],
    classifier_activation='softmax'):
  """Instantiates the ResNet101V2 architecture."""
  if not isinstance(dcn_layers, list):
    raise ValueError("dcn_layers must be a list of booleans.")

  if True in dcn_layers:
    if not file_io.file_exists_v2(weights):
      weights = None

  def stack_fn(x):
    x = resnet.stack2(x, 64, 3, use_dcn=dcn_layers[0], name='conv2')
    x = resnet.stack2(x, 128, 4, use_dcn=dcn_layers[1], name='conv3')
    x = resnet.stack2(x, 256, 23, use_dcn=dcn_layers[2], name='conv4')
    return resnet.stack2(x, 512, 3, stride1=1, use_dcn=dcn_layers[3], 
    					 name='conv5')

  return resnet.ResNet(
      stack_fn,
      True,
      True,
      'resnet101v2',
      include_top,
      weights,
      input_tensor,
      input_shape,
      pooling,
      classes,
      classifier_activation=classifier_activation)

def ResNet152V2(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    dcn_layers=[False, False, False, False],
    classifier_activation='softmax'):
  """Instantiates the ResNet152V2 architecture."""
  if not isinstance(dcn_layers, list):
    raise ValueError("dcn_layers must be a list of booleans.")

  if True in dcn_layers:
    if not file_io.file_exists_v2(weights):
      weights = None

  def stack_fn(x):
    x = resnet.stack2(x, 64, 3, use_dcn=dcn_layers[0], name='conv2')
    x = resnet.stack2(x, 128, 8, use_dcn=dcn_layers[1], name='conv3')
    x = resnet.stack2(x, 256, 36, use_dcn=dcn_layers[2], name='conv4')
    return resnet.stack2(x, 512, 3, stride1=1, use_dcn=dcn_layers[3], 
    					 name='conv5')

  return resnet.ResNet(
      stack_fn,
      True,
      True,
      'resnet152v2',
      include_top,
      weights,
      input_tensor,
      input_shape,
      pooling,
      classes,
      classifier_activation=classifier_activation)

