from typing import Sequence

from absl import app
from sight import service_utils as service
from sight.widgets.decision import decision
from sight_service.proto import service_pb2


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  req = service_pb2.TestRequest()
  response = service.call(lambda s, meta: s.Test(req, 300, metadata=meta))

  print(response)


if __name__ == "__main__":
  app.run(main)
