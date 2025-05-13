"""Test Avro schema."""

import json
import pathlib
import unittest

import fastavro
from fastavro import _validate_common
from fastavro import validation
from helpers.logs.logs_handler import logger as logging
from tests.colorful_tests import ColorfulTestRunner

ValidationError = _validate_common.ValidationError
Path = pathlib.Path
reader = fastavro.reader
writer = fastavro.writer
validate = validation.validate
parse_schema = fastavro.parse_schema


class TestAvroSchemaFromFile(unittest.TestCase):
  """Test Avro schema from a file."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    schema_path = Path(__file__).resolve().parents[2] / "avrofile-schema.avsc"
    try:
      # Load and parse the schema from the .avsc file
      with open(schema_path, "r") as f:
        raw_schema = json.load(f)
      cls.parsed_schema = parse_schema(raw_schema)
    except Exception as e:
      raise RuntimeError(
          f"Failed to parse the schema during setup : {e}") from e

  def test_is_schema_parseable(self):
    logging.info("Schema was tested in class setup function , it worked !!")

  def test_empty_record(self):
    partial_record = {}
    self.assertTrue(validate(partial_record, self.parsed_schema))

  def test_some_basic_fields(self):
    partial_record = {
        "location": "USA",
        "index": 1,
        "log_uid": "some_string",
        "attribute": [{
            "key": "some_key",
            "value": "some_value"
        }],
    }
    self.assertTrue(validate(partial_record, self.parsed_schema))

  def test_negative_test_for_basic_fields(self):
    wrong_record = {
        "location": "USA",
        "index": 1,
        "log_uid": 999,
        "attribute": [{"1", "2", "3"}],
    }
    with self.assertRaises(ValidationError):
      validate(wrong_record, self.parsed_schema)

  def test_value_fields_for_list_value(self):
    partial_record = {
        "value": {
            "sub_type": "ST_JSON",
            "list_value": [
                {
                    "sub_type": "ST_INT64",
                    "int64_value": 8988878678
                },
                {
                    "sub_type": "ST_STRING",
                    "string_value": "list"
                },
                {
                    "sub_type": "ST_JSON",
                    "list_value": [
                        {
                            "sub_type": "ST_INT64",
                            "int64_value": 123
                        },
                        {
                            "sub_type": "ST_STRING",
                            "int64_value": "sub-list"
                        },
                    ],
                },
            ],
        }
    }
    self.assertTrue(validate(partial_record, self.parsed_schema))

  def test_value_fields_for_map_value(self):
    partial_record = {
        "value": {
            "sub_type": "ST_JSON",
            "map_value": {
                "key": {
                    "sub_type": "ST_INT64",
                    "int64_value": 12345
                }
            },
        }
    }
    self.assertTrue(validate(partial_record, self.parsed_schema))

  def test_basic_propose_action_schema(self):
    partial_record = {
        "propose_action": {
            "action_id": "123",
            "action_attrs": {
                "params": {
                    "some_key1": {
                        "sub_type": "ST_STRING",
                        "string_value": "some_string",
                    }
                }
            },
        }
    }
    self.assertTrue(validate(partial_record, self.parsed_schema))

  def test_propose_action_schema_with_list_and_map(self):
    partial_record = {
        "propose_action": {
            "action_id": "123",
            "action_attrs": {
                "params": {
                    "some_key1": {
                        "sub_type": "ST_STRING",
                        "string_value": "some_string",
                    },
                    "some_key2": {
                        "sub_type": "ST_JSON",
                        "list_type": [
                            {
                                "sub_type": "ST_STRING",
                                "string_value": "some_string",
                            },
                            {
                                "sub_type": "ST_JSON",
                                "map_value": {
                                    "nested_key1": {
                                        "sub_type": "ST_STRING",
                                        "string_value": "some_string",
                                    }
                                },
                            },
                        ],
                    },
                }
            },
        }
    }
    self.assertTrue(validate(partial_record, self.parsed_schema))


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
