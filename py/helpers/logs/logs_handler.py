# import logging

# logging.basicConfig(level=logging.INFO)

# logging.info("info")
# logging.debug("debug")
# logging.warning("warning")
# logging.error("error")

import logging

from google.cloud import logging as cloud_logging

# Set this to True for Cloud logging
USE_CLOUD_LOGGING = True


class CustomAdapter(logging.LoggerAdapter):

  def process(self, msg, kwargs):
    # Include the extra context from the adapter into the log message
    extra_info = ' | '.join(
        f'{key}: {value}' for key, value in self.extra.items())
    return f'{extra_info} | {msg}', kwargs


if USE_CLOUD_LOGGING:
  logging_client = cloud_logging.Client()
  handler = logging_client.get_default_handler()
else:
  handler = logging.StreamHandler()

# Set up Python logging
logger = logging.getLogger("myLogger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
# adapter = CustomAdapter(logger, {'user': 'meetashah'})

# Example of logging
# logger.info("This is an info message logged to GCP from VM")
