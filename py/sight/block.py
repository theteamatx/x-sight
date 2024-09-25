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
"""Hierarchical blocks in the Sight log."""

import inspect
from typing import Dict, Optional, Text
from helpers.logs.logs_handler import logger as logging
from sight.exception import exception
from sight.location import Location
from sight.proto import sight_pb2
from sight.sight import Sight


class Block(object):
    """Encapsulates start and stop points where a Sight log block is active."""

    def __init__(self, *args):
        if len(args) == 2:
            (label, sight) = args
            self.create_label(label, sight)
            return
        if len(args) == 3 and isinstance(args[2], dict):
            (label, sight, attributes) = args
            self.create_label(label, sight, attributes)
            return

        if len(args) == 3:
            (key, value, sight) = args
            self.create_label('%s=%s' % (key, value), sight,
                              {str(key): str(value)})
            return
        (key, value, sight, attributes) = args
        full_attributes = attributes.copy()
        full_attributes[key] = value
        self.create_label('%s=%s' % (key, value), sight, full_attributes)

    def create_label(
        self,
        label: str,
        sight: Sight,
        attributes: Optional[Dict[Text, Text]] = None,
    ) -> Optional[Location]:
        """Creates and enters a block with a given label and attributes.

    Args:
      label: The label that identifies this block.
      sight: The logger instance to which the block is logged.
      attributes: Key-value pairs that identify this block and all of its
        contained objects.

    Returns:
      The starting location of this block.
    """
        self.sight = sight
        if sight is None:
            logging.info('<<< %s', label)
            return None

        if not self.sight.is_logging_enabled():
            return None

        self.label = label
        if attributes:
            self.attributes = attributes
        else:
            self.attributes = dict()
        for key in sorted(self.attributes.keys()):
            self.sight.set_attribute(key, self.attributes.get(key))
        # pytype: disable=attribute-error
        return self.sight.enter_block(self.label, sight_pb2.Object(),
                                      inspect.currentframe().f_back.f_back)
        # pytype: enable=attribute-error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        if not self.sight:
            return

        if not self.sight.is_logging_enabled():
            return

        if exc_type is not None:
            # pytype: disable=attribute-error
            exception(exc_type, value, traceback, self.sight,
                      inspect.currentframe().f_back)
            # pytype: enable=attribute-error

        if self.sight is None:
            logging.info('>>> %s', self.label)
            return

        # pytype: disable=attribute-error
        self.sight.exit_block(self.label, sight_pb2.Object(),
                              inspect.currentframe().f_back)
        # pytype: enable=attribute-error

        for key in sorted(self.attributes.keys(), reverse=True):
            self.sight.unset_attribute(key)
