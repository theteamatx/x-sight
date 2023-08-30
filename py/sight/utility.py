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

from google.protobuf.json_format import _Printer as BasePrinter 
from google.protobuf.json_format import SerializeToJsonError
from google.protobuf.internal import type_checkers
from google.protobuf import descriptor
from google.protobuf.json_format import _INT64_TYPES
from google.protobuf.json_format import _FLOAT_TYPES
from google.protobuf.json_format import _INFINITY
from google.protobuf.json_format import _NEG_INFINITY
from google.protobuf.json_format import _NAN


import base64
import math


def MessageToJson(
    message,
    including_default_value_fields=False,
    preserving_proto_field_name=False,
    indent=2,
    sort_keys=False,
    use_integers_for_enums=False,
    descriptor_pool=None,
    float_precision=None,
    ensure_ascii=True):
  """Converts protobuf message to JSON format.

  Args:
    message: The protocol buffers message instance to serialize.
    including_default_value_fields: If True, singular primitive fields,
        repeated fields, and map fields will always be serialized.  If
        False, only serialize non-empty fields.  Singular message fields
        and oneof fields are not affected by this option.
    preserving_proto_field_name: If True, use the original proto field
        names as defined in the .proto file. If False, convert the field
        names to lowerCamelCase.
    indent: The JSON object will be pretty-printed with this indent level.
        An indent level of 0 or negative will only insert newlines.
    sort_keys: If True, then the output will be sorted by field names.
    use_integers_for_enums: If true, print integers instead of enum names.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
        default.
    float_precision: If set, use this to specify float field valid digits.
    ensure_ascii: If True, strings with non-ASCII characters are escaped.
        If False, Unicode strings are returned unchanged.

  Returns:
    A string containing the JSON formatted protocol buffer message.
  """
  printer = _Printer(
      including_default_value_fields,
      preserving_proto_field_name,
      use_integers_for_enums,
      descriptor_pool,
      float_precision=float_precision)
  return printer.ToJsonString(message, indent, sort_keys, ensure_ascii)

def MessageToDict(
    message,
    including_default_value_fields=False,
    preserving_proto_field_name=False,
    use_integers_for_enums=False,
    descriptor_pool=None,
    float_precision=None):
  """Converts protobuf message to a dictionary.

  When the dictionary is encoded to JSON, it conforms to proto3 JSON spec.

  Args:
    message: The protocol buffers message instance to serialize.
    including_default_value_fields: If True, singular primitive fields,
        repeated fields, and map fields will always be serialized.  If
        False, only serialize non-empty fields.  Singular message fields
        and oneof fields are not affected by this option.
    preserving_proto_field_name: If True, use the original proto field
        names as defined in the .proto file. If False, convert the field
        names to lowerCamelCase.
    use_integers_for_enums: If true, print integers instead of enum names.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
        default.
    float_precision: If set, use this to specify float field valid digits.

  Returns:
    A dict representation of the protocol buffer message.
  """
  # print("descriptor_pool : ", descriptor_pool)

  printer = _Printer(
      including_default_value_fields,
      preserving_proto_field_name,
      use_integers_for_enums,
      descriptor_pool,
      float_precision=float_precision)
  # pylint: disable=protected-access
  return printer._MessageToJsonObject(message)

class _Printer(BasePrinter):
  def _FieldToJsonObject(self, field, value):
    """Converts field value according to Proto3 JSON Specification."""
    if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
      return self._MessageToJsonObject(value)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
      if self.use_integers_for_enums:
        return value
      if field.enum_type.full_name == 'google.protobuf.NullValue':
        return None
      enum_value = field.enum_type.values_by_number.get(value, None)
      if enum_value is not None:
        return enum_value.name
      else:
        if field.file.syntax == 'proto3':
          return value
        raise SerializeToJsonError('Enum field contains an integer value '
                                   'which can not mapped to an enum value.')
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
      if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
        # Use base64 Data encoding for bytes
        return base64.b64encode(value).decode('utf-8')
      else:
        return value
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
      return bool(value)
    elif field.cpp_type in _INT64_TYPES:
      # changed here so, it will not convert long type in str
      # return str(value)
      return value
    elif field.cpp_type in _FLOAT_TYPES:
      if math.isinf(value):
        if value < 0.0:
          return _NEG_INFINITY
        else:
          return _INFINITY
      if math.isnan(value):
        return _NAN
      if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_FLOAT:
        if self.float_format:
          return float(format(value, self.float_format))
        else:
          return type_checkers.ToShortestFloat(value)

    return value