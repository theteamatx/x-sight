"""Tests for proto conversion utils."""

import unittest

from sight.proto import sight_pb2
from sight.utils.proto_conversion import get_proto_value_from_value
from sight.utils.proto_conversion import get_value_from_proto_value
from tests.colorful_tests import ColorfulTestRunner


class TestGetProtoValueFromValue(unittest.TestCase):
  """Tests for get_proto_value_from_value."""

  def test_primitives(self):
    """Test that get_proto_value_from_value correctly converts primitive types.

    This test case checks the conversion of string, int64, bool, and None
    values to their corresponding sight_pb2.Value proto representations. It also
    checks the conversion of double and bytes.
    For example:
        py_val: "hello"
        expected:
            sub_type: ST_STRING
            string_value: "hello"

        py_val: 3.14
        expected:
            sub_type: ST_DOUBLE
            double_value: 3.14

        py_val: 42
        expected:
            sub_type: ST_INT64
            int64_value: 42

    The test cases iterates through a list of tuples, each containing a name,
    a Python value, and the expected sight_pb2.Value.SubType.
    """
    cases = [
        (
            "string",
            "hello",
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_STRING,
                            string_value="hello"),
        ),
        (
            "int64",
            42,
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_INT64,
                            int64_value=42),
        ),
        (
            "bool",
            True,
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_BOOL,
                            bool_value=True),
        ),
        (
            "double",
            3.14,
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_DOUBLE,
                            double_value=3.14),
        ),
        (
            "bytes",
            b"abc",
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_BYTES,
                            bytes_value=b"abc"),
        ),
    ]
    for name, py_val, expected_proto in cases:
      with self.subTest(name=name):
        val = get_proto_value_from_value(py_val)
        self.assertEqual(val, expected_proto)

  def test_list(self):
    """Test converting a list from Python value to proto.

    This test case checks the conversion of a Python list to a
    sight_pb2.Value proto with sub_type ST_JSON and a list_value.
    For example:
        py_val: [1, 2]
        expected:
            sub_type: ST_JSON
            list_value:
                values:
                    - sub_type: ST_INT64
                      int64_value: 1
                    - sub_type: ST_INT64
                      int64_value: 2
    """
    val = get_proto_value_from_value([1, 2])
    self.assertEqual(val.sub_type, sight_pb2.Value.SubType.ST_JSON)
    self.assertTrue(val.HasField("list_value"))
    self.assertEqual(len(val.list_value.values), 2)
    self.assertEqual([v.int64_value for v in val.list_value.values], [1, 2])

  def test_map(self):
    """Test converting a map from Python value to proto.

    This test case checks the conversion of a Python dictionary to a
    sight_pb2.Value proto with sub_type ST_JSON and a map_value.
    For example:
        py_val: {"x": "y"}
        expected:
            sub_type: ST_JSON
            map_value:
                fields:
                    key: "x"
                    value:
                        sub_type: ST_STRING
                        string_value: "y"
    """
    val = get_proto_value_from_value({"x": "y"})
    self.assertEqual(val.sub_type, sight_pb2.Value.SubType.ST_JSON)
    self.assertTrue(val.HasField("map_value"))
    self.assertEqual(len(val.map_value.fields), 1)
    self.assertEqual(val.map_value.fields["x"].string_value, "y")

  def test_nested(self):
    """Test converting a nested structure from Python value to proto.

    This test case checks the conversion of a nested Python structure to a
    sight_pb2.Value proto with sub_type ST_JSON. The structure is a list
    containing a map, which in turn contains a list.
    For example:
        py_val: [{"id": 1, "tags": ["a", "b"]}]
        expected:
            sub_type: ST_JSON
            list_value:
                values:
                    - sub_type: ST_JSON
                      map_value:
                        fields:
                            id:
                                sub_type: ST_INT64
                                int64_value: 1
                            tags:
                                sub_type: ST_JSON
                                list_value:
                                    values:
                                        - sub_type: ST_STRING
                                          string_value: "a"
                                        - sub_type: ST_STRING
                                          string_value: "b"
    """
    nested = [{"id": 1, "tags": ["a", "b"]}]
    val = get_proto_value_from_value(nested)
    self.assertEqual(val.sub_type, sight_pb2.Value.SubType.ST_JSON)
    nested_map = val.list_value.values[0]
    self.assertEqual(len(val.list_value.values), 1)
    self.assertEqual(nested_map.map_value.fields["id"].int64_value, 1)
    self.assertEqual(
        [
            v.string_value
            for v in nested_map.map_value.fields["tags"].list_value.values
        ],
        ["a", "b"],
    )

  def test_json_string_detection(self):
    """Test that get_proto_value_from_value correctly detects a JSON string.

    This test case checks that a string that is a valid JSON document
    is correctly detected and converted to a sight_pb2.Value with sub_type
    ST_JSON and the json_value field set to the original string.
    For example:
        py_val: '{"foo": "bar"}'
        expected:
            sub_type: ST_JSON
            json_value: '{"foo": "bar"}'

    The test case creates a string that is a valid JSON document and asserts
    that the resulting proto has the correct sub_type and json_value.
    """
    json_str = '{"foo": "bar"}'
    val = get_proto_value_from_value(json_str)
    self.assertEqual(val.sub_type, sight_pb2.Value.SubType.ST_JSON)
    self.assertEqual(val.json_value, json_str)


class TestGetValueFromProtoValue(unittest.TestCase):
  """Tests for get_value_from_proto_value."""

  def test_primitives(self):
    """Test that get_value_from_proto_value correctly converts primitive types.

    This test case checks the conversion of string, int64, bool, and null
    values from a sight_pb2.Value proto to their corresponding Python values.
    For example:
        proto_val:
            sub_type: ST_STRING
            string_value: "hello"
        expected: "hello"

        proto_val:
            sub_type: ST_INT64
            int64_value: 42
        expected: 42
    """
    cases = [
        (
            "string",
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_STRING,
                            string_value="hello"),
            "hello",
        ),
        (
            "int64",
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_INT64,
                            int64_value=42),
            42,
        ),
        (
            "bool",
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_BOOL,
                            bool_value=True),
            True,
        ),
        (
            "double",
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_DOUBLE,
                            double_value=3.14),
            3.14,
        ),
        (
            "bytes",
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_BYTES,
                            bytes_value=b"abc"),
            b"abc",
        ),
    ]
    for name, proto_val, expected in cases:
      with self.subTest(name=name):
        result = get_value_from_proto_value(proto_val)
        self.assertEqual(result, expected)

  def test_list(self):
    """Test converting a list from proto to Python value.

    The proto list contains two int64 values.
    Example:
        proto_val:
            sub_type: ST_JSON
            list_value:
                values:
                    - sub_type: ST_INT64
                      int64_value: 1
                    - sub_type: ST_INT64
                      int64_value: 2
        expected: [1, 2]
    """
    proto_val = sight_pb2.Value(
        sub_type=sight_pb2.Value.SubType.ST_JSON,
        list_value=sight_pb2.ListValue(values=[
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_INT64,
                            int64_value=1),
            sight_pb2.Value(sub_type=sight_pb2.Value.SubType.ST_INT64,
                            int64_value=2),
        ]),
    )
    result = get_value_from_proto_value(proto_val)
    self.assertEqual(result, [1, 2])

  def test_map(self):
    """Test converting a map from proto to Python value.

    This test case checks the conversion of a map (dictionary) from a
    sight_pb2.Value proto to a Python dictionary.
    For example:
        proto_val:
            sub_type: ST_JSON
            map_value:
                fields:
                    key: "a"
                    value:
                        sub_type: ST_STRING
                        string_value: "b"
        expected: {"a": "b"}

    The test case creates a sight_pb2.Value proto with a map_value containing
    a single key-value pair ("a": "b") and asserts that the converted Python
    value is a dictionary {"a": "b"}.
    """
    proto_val = sight_pb2.Value(
        sub_type=sight_pb2.Value.SubType.ST_JSON,
        map_value=sight_pb2.MapValue(
            fields={
                "a": sight_pb2.Value(
                    sub_type=sight_pb2.Value.SubType.ST_STRING,
                    string_value="b",
                )
            }),
    )
    result = get_value_from_proto_value(proto_val)
    self.assertEqual(result, {"a": "b"})

  def test_nested(self):
    """Test converting a nested structure from proto to Python value.

    This test case checks the conversion of a nested structure containing a list
    of maps, where each map has an integer 'id' and a list of strings 'tags'.

    Example:
        proto_val:
            sub_type: ST_JSON
            list_value:
                values:
                    - sub_type: ST_JSON
                      map_value:
                        fields:
                            id:
                                sub_type: ST_INT64
                                int64_value: 1
                            tags:
                                sub_type: ST_JSON
                                list_value:
                                    values:
                                        - sub_type: ST_STRING
                                          string_value: "a"
                                        - sub_type: ST_STRING
                                          string_value: "b"
        expected: [{"id": 1, "tags": ["a", "b"]}]
    """
    proto_val = sight_pb2.Value(
        sub_type=sight_pb2.Value.SubType.ST_JSON,
        list_value=sight_pb2.ListValue(values=[
            sight_pb2.Value(
                sub_type=sight_pb2.Value.SubType.ST_JSON,
                map_value=sight_pb2.MapValue(
                    fields={
                        "id": sight_pb2.Value(
                            sub_type=sight_pb2.Value.SubType.ST_INT64,
                            int64_value=1,
                        ),
                        "tags": sight_pb2.Value(
                            sub_type=sight_pb2.Value.SubType.ST_JSON,
                            list_value=sight_pb2.ListValue(values=[
                                sight_pb2.Value(
                                    sub_type=sight_pb2.Value.SubType.ST_STRING,
                                    string_value="a",
                                ),
                                sight_pb2.Value(
                                    sub_type=sight_pb2.Value.SubType.ST_STRING,
                                    string_value="b",
                                ),
                            ]),
                        ),
                    }),
            )
        ]),
    )
    result = get_value_from_proto_value(proto_val)
    self.assertEqual(result, [{"id": 1, "tags": ["a", "b"]}])


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
