
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

"""Demo for the python bindings to the Sight logging library."""

import os
from absl import app
from absl import flags
import asyncio
import numpy as np
import pandas as pd
from sight import data_structures
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight
from typing import Sequence

FLAGS = flags.FLAGS


def get_sight_instance():
  params = sight_pb2.Params(
      label='demo file',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")


async def coroutine_one(sight) -> None:
    with Block("1", sight):
      print('1 sight.location=', sight.location.get())
      await asyncio.sleep(2)
      sight.text('Coroutine one finished')

async def coroutine_two(sight) -> None:
    with Block("2", sight):
      print('2 sight.location=', sight.location.get())
      await asyncio.sleep(1)
      sight.text('Coroutine two finished')

async def main(argv: Sequence[str]) -> None:
  with get_sight_instance() as sight:
    with Block("A", sight):
      # Wrap coroutines with tasks
      task1 = sight.create_task(coroutine_one(sight), 1)
      task2 = sight.create_task(coroutine_two(sight), 2)

      # The event loop will now handle both tasks concurrently
      await task1
      await task2

def main_wrapper(argv: Sequence[str]):
  asyncio.run(main(argv))

if __name__ == "__main__":
  app.run(main_wrapper)
