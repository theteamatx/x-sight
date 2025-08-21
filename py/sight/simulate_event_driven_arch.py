import argparse
import os

from sight.sight import Sight
from sight.block import Block

# TODO: migrate to pyproject.toml


def simulate_event_driven_architecture(log_file):
    """Simulates an event-driven architecture and logs traces."""
    params = {
        'label': 'event_driven_simulation',
        'log_owner': 'test_owner',
        'local': True,
        'text_output': True,
        'log_dir_path': os.path.dirname(log_file),
        'log_file_name': os.path.basename(log_file)
    }


    with Sight(params) as sight:
        with Block('EventProcessing', sight):
            sight.text('Starting event processing simulation.')

            # Simulate receiving and processing events
            for i in range(3):
                with Block(f'Event_{i}', sight):
                    sight.text(f'Processing event {i}.')
                    # Simulate some work
                    sight.text(f'Event {i} processed successfully.')

            sight.text('Event processing simulation finished.')

    print('Event simulation complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an event-driven architecture and log traces.')
    parser.add_argument('--log_file', required=True, help='The path to the log file.')
    args = parser.parse_args()

    simulate_event_driven_architecture(args.log_file)
