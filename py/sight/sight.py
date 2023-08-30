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

from collections import defaultdict
import getpass
import inspect
import time
from typing import Any, Iterator, Optional, Sequence

from absl import logging
import grpc

import fastavro
from fastavro.schema import load_schema
from sight.gcs import uploadFileToGcsBucket,createExternalBQtable
from sight.utility import MessageToDict
from sight.proto import sight_pb2
from sight.exception import exception
from sight.location import Location
from sight.widgets.decision import decision
from sight.service import generate_metadata

import grpc
from service.decision import decision_pb2
from service.decision import decision_pb2_grpc
import subprocess
from dotenv import load_dotenv
import os
load_dotenv()
import google.auth.transport.requests
import google.oauth2.id_token

from absl import flags
from google.cloud import aiplatform_v1

def uploadFileAndCreateTable(params : sight_pb2.Params, file_name : string):
  """upload the generated file to GCS bucket and create BigQuery external table mapped to that file.

  Args:
    Object Params containing details about bucket and BigQuery Dataset.
  """
  external_file_uri =  params.external_file_uri+params.bucket_name+"/"+params.gcp_path+file_name+'*'+params.file_format
  uploadSuccess = uploadFileToGcsBucket(params.project_id, params.bucket_name, params.gcp_path, params.log_dir_path, file_name, params.file_format)
  if "PARENT_LOG_ID" not in os.environ:
    tableSuccess = createExternalBQtable(params.project_id, params.dataset_name, file_name, params.external_file_format, external_file_uri)
  else:
    tableSuccess = 0
  
  if(uploadSuccess == -1 or tableSuccess == -1):
    return -1
  return 0

class Sight(object):
  """Object that manages writing a Sight log in some structured format.

  Provides an interface for higher-level logging abstractions to be built on
  top of this base functionality.

  Attributes:
    params: The high-level configuration parameters of the logger.
    path_prefix: Prefix of the file system path where the log will be stored.
    id: Unique identifier of this log.
    in_memory_log: Indicates whether the log is being collected in-memory
      rather than being written out to the file system.
    text_log: Object that identifies the file to which the log is being written
      in text format (if it is).
    text_log_file_path: Path of the text log.
    capacitor_log: Object via which the capacitor-format log is being written
      (if it is).
    capacitor_log_file_path: Path of the capacitor log.
    avro_log: Object via which the avro-format log is being written
      (if it is).
    avro_log_file_path: Path of the avro log.
    sight_service: Stub via which the Sight service is accessed.
    channel: Channel via which the Sight service is accessed.
    pause_logging_depth: The log nesting depth beyond which logging log entries
      are not emitted.
    location: The current log location, unique within the current log. Updated
      as the log is written out.
    index: The numeric index of the current log entry, unique within the
      current log and totally ordered. Updated as the log is written out.
    line_prefix: Text prepended to the start of each emitted text-formatted log
      line.
    line_suffix: Text appended to the end of each emitted text-formatted log
      line.
    open_block_start_locations: Records the locations of all the blocks that
      are currently open. This makes it possible to connect block end points
      to their starts.
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
    citc_snapshot: The CitC snapshot id within which the binary was executed.
    widget_decision_state: The state of the decision widget, making it possible
      to carry its state across all points where the same Sight logger object
      is being used.
    widget_simulation_state: The state of the decision widget, making it
      possible to carry its state across all points where the same Sight logger
      object is being used.
  """

  # The common prefix of source code files that should be removed from emitted
  # log when documenting the logging code location.
  CODE_FILES_PATH_PREFIX = 'runfiles/google3/'

  # The absolute path of the Sight protodb file.
  PROTODB_PATH = 'google3/googlex/cortex/sight/proto2/sight_proto2db.protodb'

  # The BNS address of the Sight service
  # SIGHT_SERVICE_ADDR = '/bns/wl/borg/wl/bns/bronevet/sight-service.server/0'
  # The cloud-run service address of the Sight service
  # SIGHT_SERVICE_ADDR = 'sight-service-dq7fdwqgbq-uc.a.run.app'

  # The API Key for the Plx Sight service
  # SIGHT_API_KEY = 'AKfycbxf6dqqiVw_ebHK0wNNYSrc0Ga6YtK2zdCSseNN_H4'
  # The API Key for the BQ Sight service
  SIGHT_API_KEY = 'AKfycbz35qrsrKUmm2FITMsLW9vSbKoBxEYv4EggM_m1Q2H3'

  def __init__(self,
               params: sight_pb2.Params,
               configuration: Optional[Sequence[sight_pb2.Object]] = None):
    self.params = params

    # Initialize each widget's state to make sure its state field is created.
    self.widget_decision_state = defaultdict(dict)
    # self.widget_simulation_state = SimulationWidgetState()

    # self._configure(configuration)

    if self.params.silent_logger:
      return

    # The path prefix common to all the file(s) that hold the log.
    self.path_prefix = ''
    path_label = 'log'
    if params.label:
      path_label = params.label

    if params.in_memory:
      self.path_prefix = ''
      self.id = 0
      self.in_memory_log = []
      self.text_log = None
      self.capacitor_log = None
      self.avro_log = None
      self.file_name = params.label
    else:
      if params.local:
        self.path_prefix = '%s/%s' % (params.log_dir_path, path_label)
        self.id = 0
      
      # Added : opening Avro file
      if params.avro_output:
        logging.info('#######SERVICE###############')
        
        # # working for user account
        # with open('service/cloud_run_create.app', 'rb') as f:
        #     creds = grpc.ssl_channel_credentials(f.read())
        # channel_opts = ()
        # self.channel = grpc.secure_channel('{}:{}'.format(self.SIGHT_SERVICE_ADDR, 443), creds, channel_opts)
        # self.sight_service = decision_pb2_grpc.SightServiceStub(self.channel)
        # logging.info('##### self.sight_service=%s #####', self.sight_service)
        
        # metadata = []
        # if (os.getenv("SERVICE_ACCOUNT") == "True"):
        #   auth_req = google.auth.transport.requests.Request()
        #   id_token = google.oauth2.id_token.fetch_id_token(auth_req, 'https://'+self.SIGHT_SERVICE_ADDR)
        # else:l
        #   id_token = subprocess.getoutput(os.getenv("ID_TOKEN"))
        # metadata.append(('authorization', 'Bearer ' + id_token))

        logging.info("calling generate metadata")
        self.sight_service,metadata = generate_metadata()
        req = decision_pb2.CreateRequest(
                                            log_owner=params.log_owner,
                                            label=params.label,
                                            log_dir_path=params.log_dir_path,
                                            format='LF_AVRO'
                                        )
        
        try:
          if "PARENT_LOG_ID" in os.environ:
            logging.info("PARENT_LOG_ID found")
            worker_location = os.environ['worker_location'].replace(":","_")
            self.path_prefix = params.log_dir_path+'log'+'_'+os.environ['PARENT_LOG_ID']+'_'+worker_location
          else:
            response = self.sight_service.Create(req,  30, metadata=metadata)
            logging.info('##### response=%s #####', response)
            self.id = response.id
            logging.info("PARENT_LOG_ID not found - parent process")
            self.path_prefix = response.path_prefix+'log'+'_'+str(response.id)
        except Exception as e:
          logging.info('RPC ERROR: %s', e)
          if not params.log_dir_path:
            params.log_dir_path = '/tmp/'
          self.path_prefix = '%s/%s' % (params.log_dir_path, path_label)
          logging.exception('Logging only locally to %s due to: error %s ',
                            self.path_prefix, e)
          self.params.local = True

        self.avro_log_file_path = self.path_prefix+'.avro'
        schema_parsed = load_schema("sight/proto/avroSchema.avsc")
        datafile = open(self.avro_log_file_path, 'wb')
        self.avro_log = fastavro.write.Writer(datafile, schema_parsed)
      else:
        self.avro_log = None

      if params.text_output:
        self.text_log_file_path = self.path_prefix + '.txt'
        self.text_log = open(self.text_log_file_path, 'w')
      else:
        self.text_log = None

    self.pause_logging_depth = 0
    self.location = Location()
    if "PARENT_LOG_ID" in os.environ:
      self.location.exit()
      # self.location.enter(os.environ["PARENT_LOCATION"])
      worker_location = (os.environ["worker_location"]).split(":")
      # print("worker_location : ", worker_location)
      for loc in worker_location:
        self.location.enter(loc)
      self.location.enter(0)
      # print("self.location = ", self.location)
    self.index = 1
    self.line_prefix = ''
    self.line_suffix = ''
    self.open_block_start_locations = []
    self.num_direct_contents = Location()
    self.num_transitive_contents = Location()
    self.active_block_labels = []
    self.attributes = {}
    self.open = True

    self.set_attribute('log_uid', self.id)

    # if build_data.Changelist():
    #   self.change_list_number = int(build_data.Changelist())
    #   self.set_attribute('change_list_number', str(self.change_list_number))
    # if build_data.CitcSnapshot():
    #   self.citc_snapshot = int(build_data.CitcSnapshot())
    #   self.set_attribute('citc_snapshot', str(self.citc_snapshot))

  @classmethod
  def silent(cls) -> Sight:
    return Sight(sight_pb2.Params(silent_logger=True))

  def new(self,
          params: sight_pb2.Params,
          configuration: Optional[Sequence[sight_pb2.Object]] = None) -> Sight:
    """Returns a new instance of Sight.

    This method is useful for creating new Sight logger objects in cases where
    it is not feasible to import Sight (due to circular import dependencies)
    but there is already a dynamic Sight object from which a new log can be
    created.

    Args:
      params: Primary configuration parameters of the logger.
      configuration: Sight log that contains additional configuration details.
    """
    print("in new method")
    return Sight(params, configuration)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, value, traceback):
    if self.params.silent_logger:
      self.close()
    if exc_type is not None:
      # pytype: disable=attribute-error
      exception(exc_type, value, traceback, self, inspect.currentframe().f_back)
      # pytype: enable=attribute-error
    self.close()

  def __del__(self):
    self.close()

  def close(self):
    """Closes this logger. Finalizes all log files so are ready for use."""
    if self.params.silent_logger:
      return

    if not self.open:
      return

    # self.unset_attribute('log_uid')

    if hasattr(self, 'citc_snapshot'):
      self.unset_attribute('citc_snapshot')
    if hasattr(self, 'change_list_number'):
      self.unset_attribute('change_list_number')

    if self.text_log:
      self.text_log.close()

    if self.avro_log:
      # Wait to ensure that the server and storage infrastructure is fully
      # stable.
      time.sleep(1)
      self.avro_log.flush()
      print("self.path_prefix value here is: ", self.path_prefix)
      logging.info(self.avro_log_file_path+' file successfully generated')
      
      try:      
        logging.info('##### FINALIZING #####')
        path = self.path_prefix.split("/")
        self.file_name = path[-1]
        # write to GCS bucket and create BQ table
        uploadSuccess = uploadFileAndCreateTable(self.params, self.file_name)
        if(uploadSuccess != -1):
          logging.info(
                'Log : https://script.google.com/a/google.com/macros/s/%s/dev?'
                'log_id=%s.%s&log_owner=%s', self.SIGHT_API_KEY, self.params.dataset_name,
                self.file_name, self.params.log_owner)
      except Exception as e:
        logging.exception('Logging only locally to %s due to: error %s ',
                          self.avro_log_file_path, e)
        self.params.local = True

      
    if not self.params.local and not self.params.in_memory:
      time.sleep(1)
      try:
        logging.info('##### FINALIZING #####')
        self.sight_service.Finalize(
            decision_pb2.FinalizeRequest(
                id=self.id,
                path_prefix=self.path_prefix,
                log_owner=self.params.log_owner
                if self.params.log_owner else getpass.getuser()))
        logging.info(
            'Log : https://script.google.com/a/google.com/macros/s/%s/dev?'
            'log_id=%s&log_owner=%s', self.SIGHT_API_KEY, self.id,
            self.params.log_owner)
        logging.info(
            'gqui /cns/iq-d/home/bronevet/sight_logs/bronevet/log_%s.capacitor order by location',
            self.id)
      except grpc.RpcError as rpc_error:
        logging.exception('Logging only locally to %s due to: error %s ',
                          self.capacitor_log_file_path, rpc_error)
        self.params.local = True

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
    google3_loc = frameinfo.filename.find(self.CODE_FILES_PATH_PREFIX)
    if google3_loc >= 0:
      obj.file = frameinfo.filename[google3_loc +
                                    len(self.CODE_FILES_PATH_PREFIX):]
    else:
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
      full_text_line = '%s:%d/%s %s%s%s\n' % (obj.file, obj.line, obj.func,
                                              self.line_prefix, text_val,
                                              self.line_suffix)
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
      return self.location

    self.active_block_labels.append(label)
    self.emit_text_to_file(self.line_prefix + label + '<<<' + self.line_suffix +
                           '\n')
    self.line_prefix = self.line_prefix + label + ': '

    obj_location = self.location
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
      self.open_block_start_locations.append(obj.location)

    self.num_direct_contents.enter(0)
    self.num_transitive_contents.enter(0)
    self.location.enter(0)

    return obj_location

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

    if not self.active_block_labels or self.location.size() == 1:
      logging.warning('Exiting inactive Sight block \"%s\"', label)
      return

    self.active_block_labels.pop()
    self.line_prefix = ''
    for block_label in self.active_block_labels:
      self.line_prefix = self.line_prefix + block_label + ': '

    self.location.exit()
    self.location.next()

    if self.is_binary_logged():
      if not self.open_block_start_locations:
        logging.warning('Exiting inactive Sight block "%s"', label)

      obj.sub_type = sight_pb2.Object.SubType.ST_BLOCK_END
      if obj.block_end is None:
        obj.block_end = sight_pb2.BlockEnd()
      obj.block_end.label = label
      obj.block_end.num_direct_contents = self.num_direct_contents.pos()
      obj.block_end.num_transitive_contents = self.num_transitive_contents.pos()
      obj.block_end.location_of_block_start = self.open_block_start_locations[
          -1]
      self.open_block_start_locations.pop()

      if frame is None:
        # pytype: disable=attribute-error
        frame = inspect.currentframe().f_back
        # pytype: enable=attribute-error
      self.set_object_code_loc(obj, frame)

      self.log_object(obj, True)

    self.emit_text_to_file(self.line_prefix + label + '>>>' + self.line_suffix +
                           '\n')

    self.num_direct_contents.exit()
    self.num_transitive_contents.exit()

  def _update_line_suffix(self) -> None:
    # Each value in self.attributes is non-empty since empty values are removed
    # in unset_attribute.
    self.line_suffix = '| ' + (','.join(
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
      return self.location

    if not self.num_direct_contents.is_empty():
      self.num_direct_contents.next()
    self.num_transitive_contents.next_all()

    obj_location = self.location
    if self.is_binary_logged():
      obj.location = str(self.location)
      obj.index = self.index
      self.index += 1

      for key, value in self.attributes.items():
        if not value:
          logging.warning('No attributes recorded for key %s', key)
          continue

        attr = obj.attribute.add()
        attr.key = key
        attr.value = str(value[-1])

      for loc in self.open_block_start_locations:
        obj.ancestor_start_location.append(str(loc))
      obj.ancestor_start_location.append(str(self.location))

      obj.order.timestamp_ns = time.time_ns()

      if self.params.in_memory:
        self.in_memory_log.append(obj)
      elif self.avro_log:
        dict_obj = MessageToDict(obj, preserving_proto_field_name=True)
        self.avro_log.write(dict_obj)

    if advance_location:
      self.location.next()

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
    # return self.params.capacitor_output
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
  #     if cur.sub_type != sight_pb2.Object.ST_BLOCK_START or cur.block_start.sub_type != sight_pb2.BlockStart.ST_CONFIGURATION:
  #       continue

  #     if cur.block_start.configuration.sub_type == sight_pb2.ConfigurationStart.ST_DECISION_CONFIGURATION:
  #       decision.configure(cur.block_start.configuration.decision_configuration,
  #                          self.widget_decision_state)

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
