"""Test for Sight proto value."""

import unittest

from sight.proto import sight_pb2
from tests.colorful_tests import ColorfulTestRunner


class TestSightProtoValue(unittest.TestCase):
  """Test for Sight proto value."""

  def test_string_value(self):
    """Test for Sight proto string value."""
    val = sight_pb2.Value(sub_type=sight_pb2.Value.ST_STRING,
                          string_value="hello")
    self.assertEqual(val.sub_type, sight_pb2.Value.ST_STRING)
    self.assertEqual(val.string_value, "hello")

  def test_json_list_value(self):
    """Test for Sight proto JSON list value."""
    val = sight_pb2.Value(
        sub_type=sight_pb2.Value.ST_JSON,
        list_value=sight_pb2.ListValue(values=[
            sight_pb2.Value(sub_type=sight_pb2.Value.ST_INT64, int64_value=42),
            sight_pb2.Value(sub_type=sight_pb2.Value.ST_BOOL, bool_value=True),
            sight_pb2.Value(
                sub_type=sight_pb2.Value.ST_STRING,
                string_value="hello",
            ),
        ]),
    )

    self.assertEqual(val.sub_type, sight_pb2.Value.ST_JSON)
    self.assertTrue(val.HasField("list_value"))
    self.assertEqual(len(val.list_value.values), 3)

    self.assertEqual(val.list_value.values[0].int64_value, 42)
    self.assertTrue(val.list_value.values[1].bool_value)
    self.assertEqual(val.list_value.values[2].string_value, "hello")

  def test_json_map_value(self):
    """Test for Sight proto JSON map value."""

    val = sight_pb2.Value(
        sub_type=sight_pb2.Value.ST_JSON,
        map_value=sight_pb2.MapValue(
            fields={
                "count": sight_pb2.Value(
                    sub_type=sight_pb2.Value.ST_INT64,
                    int64_value=123456789,
                ),
                "enabled": sight_pb2.Value(sub_type=sight_pb2.Value.ST_BOOL,
                                           bool_value=True),
                "label": sight_pb2.Value(
                    sub_type=sight_pb2.Value.ST_STRING,
                    string_value="example",
                ),
            }),
    )

    self.assertEqual(val.sub_type, sight_pb2.Value.ST_JSON)
    self.assertTrue(val.HasField("map_value"))

    fields = val.map_value.fields
    self.assertIn("count", fields)
    self.assertEqual(fields["count"].int64_value, 123456789)

    self.assertIn("enabled", fields)
    self.assertTrue(fields["enabled"].bool_value)

    self.assertIn("label", fields)

    self.assertEqual(fields["label"].string_value, "example")

  def test_json_map_with_list(self):
    """Test for Sight proto JSON map with list value."""
    # Construct the list
    list_val = sight_pb2.Value(
        sub_type=sight_pb2.Value.ST_JSON,
        list_value=sight_pb2.ListValue(values=[
            sight_pb2.Value(sub_type=sight_pb2.Value.ST_INT64, int64_value=1),
            sight_pb2.Value(sub_type=sight_pb2.Value.ST_INT64, int64_value=2),
            sight_pb2.Value(sub_type=sight_pb2.Value.ST_INT64, int64_value=3),
        ]),
    )

    # Embed the list inside a map
    map_val = sight_pb2.Value(
        sub_type=sight_pb2.Value.ST_JSON,
        map_value=sight_pb2.MapValue(
            fields={
                "numbers": list_val,
                "description": sight_pb2.Value(
                    sub_type=sight_pb2.Value.ST_STRING,
                    string_value="A list of integers",
                ),
            }),
    )

    # Assertions
    self.assertEqual(map_val.sub_type, sight_pb2.Value.ST_JSON)
    self.assertTrue(map_val.HasField("map_value"))
    self.assertIn("numbers", map_val.map_value.fields)

    numbers = map_val.map_value.fields["numbers"]
    self.assertEqual(numbers.sub_type, sight_pb2.Value.ST_JSON)
    self.assertTrue(numbers.HasField("list_value"))
    self.assertEqual(len(numbers.list_value.values), 3)
    self.assertEqual(numbers.list_value.values[0].int64_value, 1)

    self.assertEqual(
        map_val.map_value.fields["description"].string_value,
        "A list of integers",
    )

  def test_list_with_map_and_nested_list(self):
    """Test for Sight proto list with map and nested list."""
    # Inner list for "tags"
    tags_list = sight_pb2.Value(
        sub_type=sight_pb2.Value.ST_JSON,
        list_value=sight_pb2.ListValue(values=[
            sight_pb2.Value(sub_type=sight_pb2.Value.ST_STRING,
                            string_value="a"),
            sight_pb2.Value(sub_type=sight_pb2.Value.ST_STRING,
                            string_value="b"),
            sight_pb2.Value(sub_type=sight_pb2.Value.ST_STRING,
                            string_value="c"),
        ]),
    )

    # Map containing "id" and "tags"
    inner_map = sight_pb2.Value(
        sub_type=sight_pb2.Value.ST_JSON,
        map_value=sight_pb2.MapValue(
            fields={
                "id": sight_pb2.Value(sub_type=sight_pb2.Value.ST_INT64,
                                      int64_value=1),
                "tags": tags_list,
            }),
    )

    # Outer list containing the map
    outer_value = sight_pb2.Value(
        sub_type=sight_pb2.Value.ST_JSON,
        list_value=sight_pb2.ListValue(values=[inner_map]),
    )

    # Assertions
    self.assertEqual(outer_value.sub_type, sight_pb2.Value.ST_JSON)
    self.assertTrue(outer_value.HasField("list_value"))
    self.assertEqual(len(outer_value.list_value.values), 1)

    map_in_list = outer_value.list_value.values[0]
    self.assertEqual(map_in_list.sub_type, sight_pb2.Value.ST_JSON)
    self.assertTrue(map_in_list.HasField("map_value"))

    fields = map_in_list.map_value.fields
    self.assertEqual(fields["id"].int64_value, 1)

    tags = fields["tags"]
    self.assertEqual(tags.sub_type, sight_pb2.Value.ST_JSON)
    self.assertTrue(tags.HasField("list_value"))
    self.assertEqual([v.string_value for v in tags.list_value.values],
                     ["a", "b", "c"])


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
