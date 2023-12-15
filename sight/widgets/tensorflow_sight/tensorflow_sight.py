# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Documentation of numpy events and data in the Sight log."""

import inspect
from typing import Any, Optional

from absl import logging
import tensorflow as tf

from proto import sight_pb2
from py.exception import exception
from py.location import Location
from py.widgets.numpy_sight import numpy_sight


class TfModelApplication(object):
  """Encapsulates start and stop points where the application of a TF model is done."""

  def __init__(self, label: str, sight: Any):
    self.sight = sight
    if not self.sight.is_logging_enabled():
      return

    self.label = label
    if sight is None:
      logging.info('<<<TFModelApplication: %s', label)
      return
    # pytype: disable=attribute-error
    self.sight.enter_block(
        self.label,
        sight_pb2.Object(
            block_start=sight_pb2.BlockStart(
                sub_type=sight_pb2.BlockStart.ST_TENSORFLOW_MODEL_APPLICATION
            )
        ),
        inspect.currentframe().f_back,
    )
    # pytype: enable=attribute-error

  def __enter__(self):
    return self

  def __exit__(self, exc_type: Any, value: Any, traceback: Any):
    if not self.sight.is_logging_enabled():
      return

    if exc_type is not None:
      # pytype: disable=attribute-error
      exception(
          exc_type, value, traceback, self.sight, inspect.currentframe().f_back
      )
      # pytype: enable=attribute-error

    if self.sight is None:
      logging.info('TFModelApplication>>> %s', self.label)
      return

    # pytype: disable=attribute-error
    self.sight.exit_block(
        self.label,
        sight_pb2.Object(
            block_end=sight_pb2.BlockEnd(
                sub_type=sight_pb2.BlockEnd.ST_TENSORFLOW_MODEL_APPLICATION
            )
        ),
        inspect.currentframe().f_back,
    )
    # pytype: enable=attribute-error


class TfModelTraining(object):
  """Encapsulates start and stop points where a TF model is trained."""

  def __init__(self, label: str, sight: Any):
    self.sight = sight
    if not self.sight.is_logging_enabled():
      return

    self.label = label
    if sight is None:
      logging.info('<<<TFModelTraining: %s', label)
      return
    # pytype: disable=attribute-error
    self.sight.enter_block(
        self.label,
        sight_pb2.Object(
            block_start=sight_pb2.BlockStart(
                sub_type=sight_pb2.BlockStart.ST_TENSORFLOW_MODEL_TRAINING
            )
        ),
        inspect.currentframe().f_back,
    )

  # pytype: enable=attribute-error

  def __enter__(self):
    return self

  def __exit__(self, exc_type: Any, value: Any, traceback: Any):
    if not self.sight.is_logging_enabled():
      return

    if exc_type is not None:
      # pytype: disable=attribute-error
      exception(
          exc_type, value, traceback, self.sight, inspect.currentframe().f_back
      )
      # pytype: enable=attribute-error

    if self.sight is None:
      logging.info('TFModelTraining>>> %s', self.label)
      return

    # pytype: disable=attribute-error
    self.sight.exit_block(
        self.label,
        sight_pb2.Object(
            block_end=sight_pb2.BlockEnd(
                sub_type=sight_pb2.BlockEnd.ST_TENSORFLOW_MODEL_TRAINING
            )
        ),
        inspect.currentframe().f_back,
    )
    # pytype: enable=attribute-error


class TfModelTrainingEpoch(object):
  """Encapsulates start and stop points of a single epoch of TF model training."""

  def __init__(self, label: str, epoch_num: int, batch_size: int, sight: Any):
    self.sight = sight
    if not self.sight.is_logging_enabled():
      return

    self.label = label
    if sight is None:
      logging.info('<<<TfModelTrainingEpoch: %s', label)
      return

    # pytype: disable=attribute-error
    self.sight.enter_block(
        self.label,
        sight_pb2.Object(
            block_start=sight_pb2.BlockStart(
                sub_type=sight_pb2.BlockStart.ST_TENSORFLOW_MODEL_TRAINING_EPOCH,
                tensor_flow_model_training_epoch=sight_pb2.TensorFlowModelTrainingEpochStart(
                    epoch_num=epoch_num, batch_size=batch_size
                ),
            )
        ),
        inspect.currentframe().f_back,
    )
    # pytype: enable=attribute-error
    self.sight.set_attribute('tensor_flow_model_epoch_body', 'false')
    self.sight.gap()
    self.sight.unset_attribute('tensor_flow_model_epoch_body')
    self.sight.set_attribute('tensor_flow_model_epoch_body', 'true')

  def __enter__(self):
    return self

  def __exit__(self, exc_type: Any, value: Any, traceback: Any):
    if not self.sight.is_logging_enabled():
      return

    if exc_type is not None:
      # pytype: disable=attribute-error
      exception(
          exc_type, value, traceback, self.sight, inspect.currentframe().f_back
      )
      # pytype: enable=attribute-error

    if self.sight is None:
      logging.info('TfModelTrainingEpoch>>> %s', self.label)
      return

    self.sight.unset_attribute('tensor_flow_model_epoch_body')
    # pytype: disable=attribute-error
    self.sight.exit_block(
        self.label,
        sight_pb2.Object(
            block_end=sight_pb2.BlockEnd(
                sub_type=sight_pb2.BlockEnd.ST_TENSORFLOW_MODEL_TRAINING_EPOCH
            )
        ),
        inspect.currentframe().f_back,
    )
    # pytype: enable=attribute-error


def log(
    label: str, tensor: tf.Tensor, sight: Any, frame: Optional[Any] = None
) -> Optional[Location]:
  """Documents a TensorFlow tensor in the Sight log if Sight is being used.

  Args:
    label: The label that identifies this array.
    tensor: The TensorFlow tensor to be logged.
    sight:  The Sight object via which logging is to be done or None if Sight is
      not being used.
    frame: The call stack frame that contains the calling context information.

  Returns:
    The location of this array in the log.
  """
  if sight is None or not sight.is_logging_enabled():
    return None

  if frame is None:
    # pytype: disable=attribute-error
    frame = inspect.currentframe().f_back
    # pytype: enable=attribute-error

  if not tf.keras.backend.is_keras_tensor(tensor) and ('numpy' in dir(tensor)):
    return numpy_sight.log(label, tensor.numpy(), sight, frame)

  return None
