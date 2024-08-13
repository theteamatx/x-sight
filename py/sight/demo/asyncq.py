from absl import app
from absl import flags
import asyncio
import itertools as it
import os
from sight import data_structures
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight
import random
import time
from typing import Sequence


_NPROD = flags.DEFINE_integer('nprod', 1, 'Number of producers.')
_NCON = flags.DEFINE_integer('ncon', 1, 'Number of consumers.')


async def makeitem(size: int = 5) -> str:
    return os.urandom(size).hex()

async def randsleep(caller, sight: Sight) -> None:
  # i = random.randint(0, 10)
  i=.1
  if caller:
    sight.text(f"{caller} sleeping for {i} seconds.")
  await asyncio.sleep(i)

async def produce(name: int, q: asyncio.Queue, sight: Sight) -> None:
  n = random.randint(0, 10)
  # n=5
  with Block('producer', name, sight):
    for item in range(n):  # Synchronous loop for each single producer
      with Block('item', item, sight):
        await randsleep(f"Producer {name}", sight)
        i = await makeitem()
        t = time.perf_counter()
        await q.put((i, t))
        sight.text(f"Producer {name} added <{i}> to queue.")
        print(f"Producer {name} added <{i}> to queue.")

async def consume(name: int, q: asyncio.Queue, sight: Sight) -> None:
  with Block('consumer', name, sight):
    while True:
      await randsleep(f"Consumer {name}", sight)
      i, t = await q.get()
      now = time.perf_counter()
      sight.text(f"Consumer {name} got element <{i}>"
            f" in {now-t:0.5f} seconds.")
      print(f"Consumer {name} got element <{i}>"
            f" in {now-t:0.5f} seconds.")
      q.task_done()

def get_sight_instance():
  params = sight_pb2.Params(
      label='demo file',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj

async def main(argv: Sequence[str]) -> None:
  with get_sight_instance() as sight:
    start = time.perf_counter()

    q = asyncio.Queue()
    producers = [sight.create_task(produce(n, q, sight)) for n in range(_NPROD.value)]
    consumers = [sight.create_task(consume(n, q, sight)) for n in range(_NCON.value)]
    await asyncio.gather(*producers)
    await q.join()  # Implicitly awaits consumers, too
    for c in consumers:
      c.cancel()

    elapsed = time.perf_counter() - start
    sight.text(f"Program completed in {elapsed:0.5f} seconds.")

def main_wrapper(argv: Sequence[str]):
  asyncio.run(main(argv))

if __name__ == "__main__":
  app.run(main_wrapper)
