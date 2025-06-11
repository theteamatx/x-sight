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
"""Core logging class that provides APIs for creating a structured log."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import contextvars
import dataclasses
import inspect
import io
import os
import random
import threading
import time
import json
from typing import Any, Optional, Sequence, Callable, Union

from absl import flags
from dotenv import load_dotenv
import fastavro
from fastavro.schema import load_schema
from helpers.logs.logs_handler import logger as logging
from sight import service_utils as service
from sight.exception import exception
from sight.gcs_utils import create_external_bq_table
from sight.gcs_utils import upload_blob_from_stream
from sight.location import Location
from sight.proto import sight_pb2
from sight.service_utils import finalize_server
from sight.utility import MessageToDict
from sight.utility import poll_network_batch_outcome
from sight.widgets.decision import decision
from sight.widgets.simulation.simulation_widget_state import (
    SimulationWidgetState)
from sight_service.proto import service_pb2
from sight_service.shared_batch_messages import DecisionMessage

load_dotenv()
_PARENT_ID = flags.DEFINE_string('parent_id', None,
                                 'Sight log Id of super script')
FLAGS = flags.FLAGS
current_script_directory = os.path.dirname(os.path.abspath(__file__))
_SCHEMA_FILE_PATH = os.path.join(current_script_directory, '..',
                                 'avrofile-schema.avsc')


def get_default_sight_params():
  """Returns a sight object with default parameters.

  If user has provided values for some of them while initializing, it will be
  used, otherwise this value are passed.
  """
  default_prams = sight_pb2.Params(
      label='default_sight',
      log_owner='bronevet@google.com',
      local=True,
      text_output=False,
      avro_output=True,
      log_dir_path='/tmp/',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
      gcp_path='sight-logs',
      file_format='.avro',
      dataset_name='sight_logs',
      external_file_format='AVRO',
      external_file_uri='gs://',
  )
  return default_prams


@dataclasses.dataclass
class SightLocationState:
  location: Location
  line_prefix: str
  line_suffix: str
  open_block_start_locations: list[Any]
  num_direct_contents: Location
  num_transitive_contents: Location
  active_block_labels: list[Any]


class Sight(object):
  """Object that manages writing a Sight log in some structured format.

  Provides an interface for higher-level logging abstractions to be built on
  top of this base functionality.

  Attributes:
    params: The high-level configuration parameters of the logger.
    path_prefix: Prefix of the file system path where the log will be stored.
    id: Unique identifier of this log.
    in_memory_log: Indicates whether the log is being collected in-memory rather
      than being written out to the file system.
    text_log: Object that identifies the file to which the log is being written
      in text format (if it is).
    text_log_file_path: Path of the text log.
    avro_log: Object via which the avro-format log is being written (if it is).
    avro_log_file_path: Path of the avro log.
    sight_service: Stub via which the Sight service is accessed.
    channel: Channel via which the Sight service is accessed.
    pause_logging_depth: The log nesting depth beyond which logging log entries
      are not emitted.
    location: The current log location, unique within the current log. Updated
      as the log is written out.
    index: The numeric index of the current log entry, unique within the current
      log and totally ordered. Updated as the log is written out.
    line_prefix: Text prepended to the start of each emitted text-formatted log
      line.
    line_suffix: Text appended to the end of each emitted text-formatted log
      line.
    open_block_start_locations: Records the locations of all the blocks that are
      currently open. This makes it possible to connect block end points to
      their starts.
    num_direct_contents: For each currently open block, the number of log
      objects directly nested within it. Updated when each object is logged.
    num_transitive_contents: For each currently open block, the number of log
      objects transitively nested within it. Updated when each object is logged.
    active_block_labels: The label of each currently open block. Updated when
      each block is entered or exited.
    attributes: The key-value attributes that describe the current log location.
    open: Indicates whether a Sight log has been opened (silent log instances
      are not).
    change_list_number: The CL number of the workspace within which the binary
      was executed.
    widget_decision_state: The state of the decision widget, making it possible
      to carry its state across all points where the same Sight logger object is
      being used.
    widget_simulation_state: The state of the decision widget, making it
      possible to carry its state across all points where the same Sight logger
      object is being used.
    avro_schema:
    avro_record_counter:
    avro_file_counter:
    file_name:
  """

  # The absolute path of the Sight protodb file.
  # PROTODB_PATH = 'google3/googlex/cortex/sight/proto2/sight_proto2db.protodb'

  # The API Key for the BQ Sight service
  # SIGHT_API_KEY = 'AKfycbz35qrsrKUmm2FITMsLW9vSbKoBxEYv4EggM_m1Q2H3' #cameltrain
  # SIGHT_API_KEY = 'AKfycbw9eY9dk-JstxeAizfMfJZ8qwHm6BVmOZEgBUey-HPL' #catan-(now generalized)
  SIGHT_API_KEY = 'AKfycbzU74yRL1Dc0Xu5--oJricaD-H50UgF3FKM_E8_CMP7uNesQEk-k3cm57R3vTsjbWCcxA'

  @classmethod
  def create(cls, params: Union[dict, sight_pb2.Params], config=None) -> Sight:
    if isinstance(params, dict):
        try:
            params = sight_pb2.Params(**params)
        except TypeError as e:
            raise ValueError(f"Invalid params for sight_pb2.Params: {e}")
    elif not isinstance(params, sight_pb2.Params):
        raise ValueError("Expected params to be a dict or sight_pb2.Params instance.")

    return Sight(params, config)

  def _initialize_default_params(self, params: sight_pb2.params):
    """get default parameter and merges user-provided values."""
    # get default params to run sight
    default_params = get_default_sight_params()
    # replacing fields provided by user
    default_params.MergeFrom(params)
    default_params.label = default_params.label.replace(' ', '_')
    self.params = default_params

  def _load_worker_location(self):
    if 'PARENT_LOG_ID' in os.environ:
      self.location.get().exit()
      worker_location = (os.environ['worker_location']).split(':')
      for loc in worker_location:
        self.location.get().enter(loc)
      self.location.get().enter(0)

  def _initialize_context_vars(self):
    self.location = contextvars.ContextVar('location', default=Location())
    self.line_prefix = contextvars.ContextVar('line_prefix', default='')
    self.line_suffix = contextvars.ContextVar('line_suffix', default='')
    self.open_block_start_locations = contextvars.ContextVar(
        'open_block_start_locations', default=[])
    self.num_direct_contents = contextvars.ContextVar('num_direct_contents',
                                                      default=Location())
    self.num_transitive_contents = contextvars.ContextVar(
        'num_transitive_contents', default=Location())
    self.active_block_labels = contextvars.ContextVar('active_block_labels',
                                                      default=[])
    self.active_block_start_time = contextvars.ContextVar(
        'active_block_start_time', default=[])
    self.active_block_deeper_elapsed_time = contextvars.ContextVar(
        'active_block_deeper_elapsed_time', default=[0])

  def _initialize_logging(self):
    self.index = 1
    self.attributes = {}
    self.open = True
    self.id = 0
    self.set_attribute('log_uid', str(self.id))
    # Configure the tracking state of the Sight object, which records the current location
    # in the log of the current task, including its hierarchical nesting.
    self.pause_logging_depth = 0

    # The path prefix common to all the file(s) that hold the log.
    self.path_prefix = ''
    self.path_label = 'log'
    if self.params.label:
      self.path_label = self.params.label

    if self.params.local:
      self.path_prefix = '%s/%s' % (self.params.log_dir_path, self.path_label)
      self.id = 0

  def _link_child_to_parent(self):
    """ link existing log to it's parent log id with relationship as
        child to parent
    """
    if (FLAGS.parent_id):
      sight_obj = sight_pb2.Object()
      sight_obj.sub_type = sight_pb2.Object.SubType.ST_LINK
      sight_obj.link.linked_sight_id = FLAGS.parent_id
      sight_obj.link.link_type = sight_pb2.Link.LinkType.LT_CHILD_TO_PARENT
      frame = inspect.currentframe().f_back.f_back.f_back
      self.set_object_code_loc(sight_obj, frame)
      self.log_object(sight_obj, True)

  def _set_log_id_and_path_prefix(self):
    if 'PARENT_LOG_ID' in os.environ:
      logging.info('PARENT_LOG_ID found - worker process')
      worker_location = os.environ['worker_location'].replace(':', '_')
      self.path_prefix = (self.params.label + '_' +
                          os.environ['PARENT_LOG_ID'] + '_' + 'worker' + '_' +
                          worker_location + '_' + 'log')
      self.id = os.environ['PARENT_LOG_ID']
      print("log id is : ", self.id)
    elif (FLAGS.sight_log_id):
      logging.info('Using provided sight id')
      self.id = FLAGS.sight_log_id
      self.path_prefix = (self.params.label + '_' + self.id + '_' + 'log' +
                          '_run_mode')
    else:
      req = service_pb2.CreateRequest()
      response = service.call(lambda s, meta: s.Create(req, 300, metadata=meta))
      # logging.info('##### response=%s #####', response)
      self.id = response.id
      # logging.info('PARENT_LOG_ID not found - parent process')
      self.path_prefix = (self.params.label + '_' + str(response.id) + '_' +
                          'log')

  def _initialize_avro_output(self):
    # Added : opening Avro file
    if self.params.avro_output:
      self.avro_log = io.BytesIO()
      self.avro_record_counter = 0
      self.avro_file_counter = 0

      if 'SIGHT_PATH' in os.environ:
        self.avro_schema = load_schema(
            f'{os.environ["SIGHT_PATH"]}/../avrofile-schema.avsc')
      else:
        self.avro_schema = load_schema(_SCHEMA_FILE_PATH)

      try:
        self._link_child_to_parent()
        self._set_log_id_and_path_prefix()
      except Exception as e:
        logging.info('RPC ERROR: %s', e)
        if not self.params.log_dir_path:
          self.params.log_dir_path = '/tmp/'
        self.path_prefix = '%s/%s' % (self.params.log_dir_path, self.path_label)
        logging.exception(
            'Logging only locally to %s due to: error %s ',
            self.path_prefix,
            e,
        )
        self.params.local = True

      self.avro_log_file_path = (self.params.label + '_' + str(self.id) + '/' +
                                 self.path_prefix)
      self.file_name = self.avro_log_file_path.split('/')[-1]
      self.table_name = self.params.label + '_' + str(self.id) + '_' + 'log'

  def _initialize_text_output(self):
    if self.params.text_output:
      self.text_log_file_path = self.path_prefix + '.txt'
      self.text_log = open(self.text_log_file_path, 'w')
    else:
      self.text_log = None

  def __init__(
      self,
      params: sight_pb2.Params,
      config: Optional[decision.DecisionConfig] = None,
  ):
    self._initialize_default_params(params)

    # Initialize each widget's state to make sure its state field is created.
    self.widget_decision_state = defaultdict(dict)
    self.widget_simulation_state = SimulationWidgetState()

    self._initialize_context_vars()
    self._load_worker_location()
    self._initialize_logging()

    if self.params.silent_logger:
      return

    self._initialize_avro_output()
    self._initialize_text_output()

    # config should only be passed from client script, workers should never enter
    # this condition
    if config.questions and FLAGS.worker_mode is None:
      decision.initialize(config, self)

  def get_location_state(self) -> SightLocationState:
    return SightLocationState(
        self.location.get().clone(),
        self.line_prefix.get(),
        self.line_suffix.get(),
        self.open_block_start_locations.get().copy(),
        self.num_direct_contents.get().clone(),
        self.num_transitive_contents.get().clone(),
        self.active_block_labels.get().copy(),
    )

  def set_location_state(self, state: SightLocationState) -> None:
    self.location.set(state.location)
    self.line_prefix.set(state.line_prefix)
    self.line_suffix.set(state.line_suffix)
    self.open_block_start_locations.set(state.open_block_start_locations)
    self.num_direct_contents.set(state.num_direct_contents)
    self.num_transitive_contents.set(state.num_transitive_contents)
    self.active_block_labels.set(state.active_block_labels)

  def create_task(self, func):
    frame = inspect.currentframe().f_back

    async def go(func, state):
      self.set_location_state(state)
      return await func

    self.enter_block(
        f'asyncio.create_task: {asyncio.current_task().get_name()}',
        sight_pb2.Object(), frame)
    state = self.get_location_state()

    new_task = asyncio.create_task(go(func, state))  #, name=f'task_{task_id}')
    self.exit_block(f'asyncio.create_task: {asyncio.current_task().get_name()}',
                    sight_pb2.Object(), frame)
    return new_task

  @classmethod
  def silent(cls) -> Sight:
    return Sight(sight_pb2.Params(silent_logger=True))

  def new(
      self,
      params: sight_pb2.Params,
      configuration: Optional[Sequence[sight_pb2.Object]] = None,
  ) -> Sight:
    """Returns a new instance of Sight.

    This method is useful for creating new Sight logger objects in cases where
    it is not feasible to import Sight (due to circular import dependencies)
    but there is already a dynamic Sight object from which a new log can be
    created.

    Args:
      params: Primary configuration parameters of the logger.
      configuration: Sight log that contains additional configuration details.
    """
    return Sight(params, configuration)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, value, traceback):
    if self.params.silent_logger:
      self.close()
      return
    if exc_type is not None:
      # pytype: disable=attribute-error
      exception(exc_type, value, traceback, self, inspect.currentframe().f_back)
      # pytype: enable=attribute-error

    # last rpc call to server for this sight id
    req = service_pb2.CloseRequest()
    req.client_id = str(self.id)
    if 'PARENT_LOG_ID' in os.environ:
      req.question_label = self.params.label
    response = service.call(lambda s, meta: s.Close(req, 300, metadata=meta))
    # print("close rpc status :", response.response_str)
    self.close()

  def __del__(self):
    self.close()

  def _close_avro_log(self):
    self.avro_file_counter += 1
    upload_blob_from_stream(
        self.params.bucket_name,
        self.params.gcp_path,
        self.avro_log,
        self.avro_log_file_path,
        self.avro_file_counter,
    )
    # if this is the only avro file, table has not been created yet
    # if sight_log_id flags already set, table is created already
    if (self.avro_file_counter == 1 and (not FLAGS.sight_log_id)):
      create_external_bq_table(self.params, self.table_name, self.id)
    print('Log GUI : https://script.google.com/a/google.com/macros/s/%s/exec?'
          'log_id=%s.%s&log_owner=%s&project_id=%s' %
          (self.SIGHT_API_KEY, self.params.dataset_name, self.table_name,
           self.params.log_owner, os.environ['PROJECT_ID']))
    print(f'table generated : {self.params.dataset_name}.{self.table_name}')
    self.avro_log.close()

  def close(self):
    """Closes this logger. Finalizes all log files so are ready for use."""
    if self.params.silent_logger:
      return

    if not self.open:
      return

    if self.text_log:
      self.text_log.close()

    if self.avro_log and self.avro_log.getbuffer().nbytes > 0:
      self._close_avro_log()

    #? need to check whether we need this condition at all?
    if not self.params.local and not self.params.in_memory:
      print('Log GUI : https://script.google.com/a/google.com/macros/s/%s/exec?'
          'log_id=%s.%s&log_owner=%s&project_id=%s' %
          (self.SIGHT_API_KEY, self.params.dataset_name, self.table_name,
           self.params.log_owner, os.environ['PROJECT_ID']))

    if (FLAGS.decision_mode == 'train'):
      decision.finalize(self)
    finalize_server()
    self.open = False

  def pause_logging(self) -> None:
    self.pause_logging_depth += 1

  def resume_logging(self) -> None:
    self.pause_logging_depth -= 1

  def is_logging_enabled(self) -> bool:
    return not self.params.silent_logger and self.pause_logging_depth <= 1

  def get_in_memory_log(self) -> sight_pb2.Log:
    """Returns a proto that contains the full Sight in-memory log."""
    log = sight_pb2.Log()

    if self.in_memory_log:
      log.obj.extend(self.in_memory_log)

    return log

  def set_object_code_loc(self, obj: sight_pb2.Object, frame: Any) -> None:
    """Updates obj with the calling context information in frame.

    Args:
      obj: The object to be updated
      frame: The call stack frame that contains the calling context information.
    """

    frameinfo = inspect.getframeinfo(frame)
    obj.file = frameinfo.filename
    obj.line = frameinfo.lineno
    obj.func = frameinfo.function

  def text(self, text_val: str, end='\n', frame=None) -> str:
    """Logs a text value to the Sight log.

    Args:
      text_val: The text value to be logged.
      end: the character to print at the end of the text segment.
      frame: The call stack frame that the calling context from which the
        logging event was created.

    Returns:
      The logged text.
    """
    if self.params.silent_logger or self.pause_logging_depth > 0:
      return ''

    obj = sight_pb2.Object()
    if frame is None:
      # pytype: disable=attribute-error
      frame = inspect.currentframe().f_back
      # pytype: enable=attribute-error
    self.set_object_code_loc(obj, frame)

    if self.is_binary_logged():
      obj.sub_type = sight_pb2.Object.SubType.ST_TEXT
      obj.text.text = text_val.replace('\n', '\\n') + end
      obj.text.sub_type = sight_pb2.Text.SubType.ST_TEXT
      self.log_object(obj, True)

    if end == '\n':
      full_text_line = '(%s:%d) function : %s\n %s\n' % (
          obj.file,
          obj.line,
          obj.func,
          # self.line_prefix,
          text_val,
          # self.line_suffix,
      )
    else:
      full_text_line = text_val + end
    self.emit_text_to_file(full_text_line)

    return full_text_line

  def text_block(self, label: str, text_val: str, end='\n', frame=None) -> str:
    """Logs a block that contains a specified text string as its contents.

    Args:
      label: The label of the block.
      text_val: The text line to be logged.
      end: the character to print at the end of the text segment.
      frame: The call stack frame that the calling context from which the
        logging event was created.

    Returns:
      The logged text.
    """
    if self.params.silent_logger or self.pause_logging_depth > 0:
      return ''

    if frame is None:
      # pytype: disable=attribute-error
      frame = inspect.currentframe().f_back
      # pytype: enable=attribute-error
    self.enter_block(label, sight_pb2.Object(), frame)
    ret_val = self.text(text_val, end, frame)
    self.exit_block(label, sight_pb2.Object(), frame)

    return ret_val

  def gap(self) -> Optional[Location]:
    """Logs a dummy gap value value to the Sight log.

    Returns:
      The location of the dummy object in the log.
    """
    if self.params.silent_logger or self.pause_logging_depth > 0:
      return None

    if self.is_binary_logged():
      return self.log_object(
          sight_pb2.Object(sub_type=sight_pb2.Object.SubType.ST_GAP), True)

  def enter_block(self,
                  label: str,
                  obj: sight_pb2.Object,
                  frame: Optional[Any] = None) -> Optional[Location]:
    """Documents in the Sight log that a hierarchical block was entered.

    Args:
      label: The label of the block.
      obj: A Sight object where the block's entry information is to be recorded.
        This object may contain additional information that describes a custom
        log object with block-structured semantics.
      frame: The call stack frame that the calling context from which the
        logging event was created.

    Returns:
      The log Location of the block's starting point.
    """
    if self.params.silent_logger:
      return None

    if self.pause_logging_depth > 0:
      return self.location.get()

    self.active_block_labels.get().append(label)
    self.active_block_start_time.get().append(time.time_ns())
    self.active_block_deeper_elapsed_time.get().append(0)

    # self.emit_text_to_file(
    #     self.line_prefix + label + '<<<' + self.line_suffix + '\n'
    # )
    self.emit_text_to_file(self.line_prefix.get() + label + '\n' + '>>> ' +
                           '\n')
    self.line_prefix.set(self.line_prefix.get() + label + ': ')

    obj_location = self.location.get()
    if self.is_binary_logged():
      obj.sub_type = sight_pb2.Object.SubType.ST_BLOCK_START

      if obj.block_start is None:
        obj.block_start = sight_pb2.BlockStart()
      obj.block_start.label = label

      if frame is None:
        # pytype: disable=attribute-error
        frame = inspect.currentframe().f_back
      # pytype: enable=attribute-error
      self.set_object_code_loc(obj, frame)

      self.log_object(obj, False)
      self.open_block_start_locations.get().append(obj.location)

    self.num_direct_contents.get().enter(0)
    self.num_transitive_contents.get().enter(0)
    self.location.get().enter(0)

    return obj_location

  def _populate_block_end_metrics(self, obj: sight_pb2.Object,
                                  label: str) -> None:
    """Populates block_end metrics like elapsed time and exclusive time."""
    if not self.open_block_start_locations.get():
      logging.warning('Exiting inactive Sight block "%s"', label)

    obj.sub_type = sight_pb2.Object.SubType.ST_BLOCK_END
    if obj.block_end is None:
      obj.block_end = sight_pb2.BlockEnd()
    obj.block_end.label = label
    obj.block_end.num_direct_contents = self.num_direct_contents.get().pos()
    obj.block_end.num_transitive_contents = self.num_transitive_contents.get(
    ).pos()
    obj.block_end.location_of_block_start = self.open_block_start_locations.get(
    )[-1]

    elapsed_time_ns = time.time_ns() - self.active_block_start_time.get()[-1]
    obj.block_end.metrics.elapsed_time_ns = elapsed_time_ns
    obj.block_end.metrics.exclusive_elapsed_time_ns = elapsed_time_ns - self.active_block_deeper_elapsed_time.get(
    )[-1]

    self.active_block_deeper_elapsed_time.get().pop()
    self.active_block_deeper_elapsed_time.get()[-1] += elapsed_time_ns
    obj.block_end.metrics.elapsed_time_ns = elapsed_time_ns

    self.open_block_start_locations.get().pop()
    self.active_block_start_time.get().pop()

  def exit_block(self, label: str, obj: sight_pb2.Object, frame=None) -> None:
    """Documents in the Sight log that a hierarchical block was exited.

    Args:
      label: the label of the block.
      obj: a Sight object where the block's exit information is to be recorded.
        This object may contain additional information that describes a custom
        log object with block-structured semantics.
      frame: the call stack frame that the calling context from which the
        logging event was created.
    """
    if self.params.silent_logger or self.pause_logging_depth > 0:
      return

    if not self.active_block_labels.get() or self.location.get().size() == 1:
      logging.warning('Exiting inactive Sight block "%s"', label)
      return

    self.active_block_labels.get().pop()
    self.line_prefix.set('')
    for block_label in self.active_block_labels.get():
      self.line_prefix.set(self.line_prefix.get() + block_label + ': ')

    self.location.get().exit()
    self.location.get().next()

    if self.is_binary_logged():
      self._populate_block_end_metrics(obj, label)

      if frame is None:
        # pytype: disable=attribute-error
        frame = inspect.currentframe().f_back
        # pytype: enable=attribute-error
      self.set_object_code_loc(obj, frame)

      self.log_object(obj, True)

    self.emit_text_to_file(
        # self.line_prefix + label + '>>>' + self.line_suffix + '\n'
        '<<< ' + '\n')

    self.num_direct_contents.get().exit()
    self.num_transitive_contents.get().exit()

  def _update_line_suffix(self) -> None:
    # Each value in self.attributes is non-empty since empty values are removed
    # in unset_attribute.
    self.line_suffix.set('| ' + ','.join(
        [f'{key}={value[-1]}' for key, value in self.attributes.items()]))

  def set_attribute(self, key: str, value: str) -> None:
    """Documents in the Sight log a new key-value attribute mapping.

    Until the mapping is unset all logged objects will be annotated with this
    key/value pair.

    Args:
      key: the name of the key being set.
      value: the value assigned to key.
    """
    self.attributes.setdefault(key, []).append(value)
    self._update_line_suffix()

  def unset_attribute(self, key: str) -> None:
    """Removes from the Sight log a new key-value attribute mapping.

    Subsequent logged logged objects will no longer be annotated with this
    key/value pair. If the key had a value mapped to it before the value
    it was most recently set to, after this call the key will be mapped to
    that most recent value.

    Args:
      key: the name of the key being unset.
    """
    values = self.attributes.get(key)
    if not values:
      logging.error('Failed to unset attribute %s, which is not set.', key)
      return

    values.pop()
    if not values:
      del self.attributes[key]

    self._update_line_suffix()

  def fetch_attributes(self) -> dict[str, str]:
    """Fetches all the values of attributes that is currently set to within Sight.
    Returns:
      The dictionary that contains key-value pairs of attributes currently set to.
    """
    attr_dict = {}
    for k, v in self.attributes.items():
      attr_dict[k] = v[-1]
    return attr_dict

  def get_attribute(self, key: str) -> str:
    """Fetches the value that a key is currently set to within Sight.

    Args:
      key: the name of the key being fetched.

    Returns:
      The value that key is currently set to.
    """
    values = self.attributes.get(key)
    if not values:
      return ''
    return values[-1]

  def _upload_avro_file_to_gcs(self):
    # logging.info('_upload_avro_file_to_gcs self.avro_file_counter=%s', self.avro_file_counter)
    self.avro_file_counter += 1
    upload_blob_from_stream(
        self.params.bucket_name,
        self.params.gcp_path,
        self.avro_log,
        self.avro_log_file_path,
        self.avro_file_counter,
    )
    # if sight_log_id flags already set, table is created already
    if (self.avro_file_counter == 1 and (not FLAGS.sight_log_id)):
      create_external_bq_table(self.params, self.table_name, self.id)
      print(
          'Log GUI : https://script.google.com/a/google.com/macros/s/%s/exec?'
          'log_id=%s.%s&log_owner=%s&project_id=%s' % (self.SIGHT_API_KEY,
          self.params.dataset_name, self.table_name, self.params.log_owner,
          os.environ['PROJECT_ID']))
    self.avro_log.close()
    self.avro_log = io.BytesIO()

  def _write_avro_file(self, obj):
    dict_obj = MessageToDict(obj, preserving_proto_field_name=True)
    fastavro.writer(self.avro_log, self.avro_schema, [dict_obj])
    self.avro_record_counter += 1
    if self.avro_record_counter % 1000 == 0:
      self._upload_avro_file_to_gcs()

  def _flush_log(self):
    """Flushes the log to remote storage."""
    if self.avro_log:
      self._upload_avro_file_to_gcs()

  def log_object(self,
                 obj: sight_pb2.Object,
                 advance_location: bool = True) -> Optional[Location]:
    """Emits a single object to the Sight log.

    Args:
      obj: A Sight object where log event is to be recorded. This object may
        contain additional information that describes a custom log object.
      advance_location: Indicates whether this method call should advance the
        current log location.

    Returns:
      The Location of the logged object.
    """
    if self.params.silent_logger:
      return None

    if self.pause_logging_depth > 0:
      return self.location.get()

    if not self.num_direct_contents.get().is_empty():
      self.num_direct_contents.get().next()
    self.num_transitive_contents.get().next_all()

    obj_location = self.location.get()
    if self.is_binary_logged():
      obj.location = str(self.location.get())
      obj.index = self.index
      self.index += 1

      for key, value in self.attributes.items():
        if not value:
          logging.warning('No attributes recorded for key %s', key)
          continue

        attr = obj.attribute.add()
        attr.key = key
        attr.value = str(value[-1])

      for loc in self.open_block_start_locations.get():
        obj.ancestor_start_location.append(str(loc))
      obj.ancestor_start_location.append(str(self.location.get()))

      obj.order.timestamp_ns = time.time_ns()

      if self.params.in_memory:
        self.in_memory_log.append(obj)
      elif self.avro_log:
        self._write_avro_file(obj)

    if advance_location:
      self.location.get().next()

    return obj_location

  def emit_text_to_file(self, text_val: str) -> None:
    """Emits text to the output text file, if one is being used.

    Args:
      text_val: The text to be logged.
    """
    if self.params.silent_logger or self.pause_logging_depth > 0:
      return

    if self.text_log:
      self.text_log.write(text_val)
    # logging.info(text_val)

  def is_binary_logged(self) -> bool:
    """Returns whether a binary proto representation is being logged."""
    return self.params.avro_output

  def _configure(self, configuration: Sequence[sight_pb2.Object]) -> None:
    """Initializes the configuration of this logger and widgets.

    Args:
      configuration: Sight log that stores configuration log objects.
    """
    if not configuration:
      decision.configure(None, self.widget_decision_state)
      return

    self.add_config(configuration)

  # def add_config(self, configuration: Sequence[sight_pb2.Object]) -> None:
  #   """Augments the configuration of this logger from an in-memory log.

  #   Args:
  #     configuration: Sight log that stores configuration log objects.
  #   """
  #   if not configuration:
  #     return
  #   for cur in configuration:
  #     if (
  #         cur.sub_type != sight_pb2.Object.ST_BLOCK_START
  #         or cur.block_start.sub_type != sight_pb2.BlockStart.ST_CONFIGURATION
  #     ):
  #       continue

  #     if (
  #         cur.block_start.configuration.sub_type
  #         == sight_pb2.ConfigurationStart.ST_DECISION_CONFIGURATION
  #     ):
  #       decision.configure(
  #           cur.block_start.configuration.decision_configuration,
  #           self.widget_decision_state,
  #       )

  # def add_config_file(self, config_file_path: str) -> None:
  #   """Augments the configuration of this logger from a file.

  #   Args:
  #     config_file_path: File glob that contains a Sight log that stores
  #       configuration log objects.
  #   """
  #   self.add_config(_read_capacitor_file(config_file_path))  # pytype: disable=wrong-arg-types  # dynamic-method-lookup


def text(text_val: str, sight, end='\n', frame=None) -> str:
  """Logs a text value to the Sight log if Sight is being used.

  If no Sight logger object is provided, nothing is logged.

  Args:
    text_val: The text value to be logged.
    sight: The Sight object via which logging is to be done or None if Sight is
      not being used.
    end: the character to print at the end of the text segment.
    frame: The call stack frame that contains the calling context information.

  Returns:
    The logged text.
  """
  if sight.params.silent_logger or sight.pause_logging_depth > 0:
    return ''

  if sight is None:
    return ''

  if frame is None:
    # pytype: disable=attribute-error
    frame = inspect.currentframe().f_back
    # pytype: enable=attribute-error
  return sight.text(text_val, end=end, frame=frame)


def text_block(label: str, text_val: str, sight, frame=None) -> str:
  """Logs to Sight a block that contains a text string if Sight is being used.

  If no Sight logger object is provided, nothing is logged.

  Args:
    label: The label of the block.
    text_val: The text value to be logged,
    sight: The Sight object via which logging is to be done or None if Sight is
      not being used.
    frame: The call stack frame that contains the calling context information.

  Returns:
    The logged text.
  """
  if sight.params.silent_logger or sight.pause_logging_depth > 0:
    return ''

  if sight is None:
    return ''

  if frame is None:
    # pytype: disable=attribute-error
    frame = inspect.currentframe().f_back
    # pytype: enable=attribute-error
  return sight.text_block(label, text_val, frame)


def run_worker(
  driver_fn:  Callable[[Any], Any] = None,
  sight_params: dict = None,
):
  """Wrapped the driver function with decision API,
     One can directly call run_generic_worker function,
     if have their own driver function.
  """
  def wrapped_driver_fn(sight):
    action = decision.decision_point(sight_params['label'], sight)
    reward, outcome = driver_fn(action)
    decision.decision_outcome('decisionin_outcome', sight, reward, outcome)
  return run_generic_worker(wrapped_driver_fn, sight_params)

def run_generic_worker(
    driver_fn: Optional[Callable[[Any], Any]] = None,
    sight_params: dict = None,
):
  """Generic worker utility for running applications that use the Decision API.
  """

  sight = Sight.create(sight_params)
  sight.widget_decision_state['num_decision_points'] = 0

  optimizer = decision.Optimizer()
  optimizer.obj = decision.setup_optimizer(sight, FLAGS.optimizer_type)
  client_id, worker_location = decision._configure_client_and_worker(
      sight=sight)
  num_retries = 0
  backoff_interval = 0.5
  while True:
    # #? new rpc just to check move forward or not?

    req = service_pb2.WorkerAliveRequest(
        client_id=client_id,
        worker_id=f'client_{client_id}_worker_{worker_location}',
        question_label=sight_params['label'])
    response = service.call(
        lambda s, meta: s.WorkerAlive(req, 300, metadata=meta))
    logging.info('Response from WorkerAlive RPC: %s', response)
    if (response.status_type ==
        service_pb2.WorkerAliveResponse.StatusType.ST_DONE):
      break
    elif (response.status_type ==
          service_pb2.WorkerAliveResponse.StatusType.ST_RETRY):
      # logging.info('Retrying in 5 seconds......')
      # time.sleep(5)
      backoff_interval *= 2
      time.sleep(random.uniform(backoff_interval / 2, backoff_interval))
      logging.info('backed off for %s seconds... and trying for %s',
                   backoff_interval, num_retries)
      num_retries += 1
      if (num_retries >= 50):
        break
    elif (response.status_type ==
          service_pb2.WorkerAliveResponse.StatusType.ST_ACT):
      process_worker_action(response, sight, driver_fn, sight_params['label'],
                            optimizer.obj)
    else:
      raise ValueError('Invalid response from server')

    logging.info('Exiting the training loop.')

  logging.debug('<<<<<< Exiting run method')


def process_worker_action(response, sight, driver_fn, question_label, opt_obj):
  """Processes worker actions during local training.

  Args:
      response: The response from the WorkerAlive RPC.
      sight: Sight object used for logging and configuration.
      driver_fn: The driver function that drives the training.
      env: The environment in which the training takes place (optional).
      question_label:
  """
  decision_messages = decision.get_decision_messages_from_proto(
      decision_messages_proto=response.decision_messages)
  # shared_batch_messages = CachedBatchMessages()
  sight.widget_decision_state['cached_messages'] = opt_obj.cache
  logging.info('cached_messages=%s',
               sight.widget_decision_state['cached_messages'])

  for action_id, action_params in decision_messages.items():
    logging.info('action_id=%s, action_params=%s', action_id, action_params)
    sight.enter_block('Decision Sample', sight_pb2.Object())

    if 'constant_action' in sight.widget_decision_state:
      del sight.widget_decision_state['constant_action']

    cached_messages = sight.widget_decision_state['cached_messages']
    sight.widget_decision_state['discount'] = 0
    sight.widget_decision_state['last_reward'] = None
    sight.widget_decision_state['action_id'] = action_id

    cached_messages.set(
        action_id,
        DecisionMessage(
            action_id=action_id,
            action_params=action_params,
        ),
    )

    driver_fn(sight)
    sight._flush_log()

    sight.exit_block('Decision Sample', sight_pb2.Object())

  decision.finalize_episode(sight, question_label, opt_obj)
