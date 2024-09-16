# import logging

# logging.basicConfig(level=logging.INFO)

# logging.info("info")
# logging.debug("debug")
# logging.warning("warning")
# logging.error("error")





import logging
from google.cloud import logging as cloud_logging

# Instantiates a client
logging_client = cloud_logging.Client()

# Retrieves a Cloud Logging handler based on the environment
# you're running in and integrates the handler with the
# Python logging module. By default this captures all logs
# at INFO level and higher
handler = logging_client.get_default_handler()

# Set up Python logging
logger = logging.getLogger("cloudLogger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Example of logging
# logger.info("This is an info message logged to GCP")

