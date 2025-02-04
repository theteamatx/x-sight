import os
from typing import Sequence

from absl import app
from sight import service_utils as service
from helpers.logs.logs_handler import logger as logging


project_name = 'cameltrain'
deployment_mode = 'worker_mode'
worker_mode = 'dsub_local_worker'
remote_script = 'py/sight/demo/test_rpc.py'
docker_image = 'gcr.io/cameltrain/test-dsub_local'
sight_id = '1234'

def main(argv: Sequence[str]) -> None:
  script_args = (
      f'--deployment_mode={deployment_mode} --worker_mode={worker_mode} '
  )

  args = [
      'dsub',
      '--provider=local',
      f'--image={docker_image}',
      f'--project={project_name}',
      f'--logging=extra/dsub-logs',
      '--env',
      f'GOOGLE_CLOUD_PROJECT={os.environ["PROJECT_ID"]}',
      '--env',
      f'PROJECT_ID={project_name}',
      '--env',
      f'PARENT_LOG_ID={sight_id}',
      # '--env',
      # f'SIGHT_SERVICE_ID={service._SERVICE_ID}',
      '--input',
      f'SCRIPT={remote_script}',
      '--input-recursive',
      f'CLOUDSDK_CONFIG={os.path.expanduser("~")}/.config/gcloud',
      f'--command=python3 "${{SCRIPT}}" {script_args}',
      # '--tasks',
      # '/tmp/optimization_tasks.tsv',
      '--name',
      'test1234',
    ]
  logging.info('CLI=%s', ' '.join(args))
  # subprocess.run(args, check=True)


if __name__ == "__main__":
  app.run(main)
