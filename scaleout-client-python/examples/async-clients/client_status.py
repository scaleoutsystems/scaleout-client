"""This script monitors and records the status of Scaleout clients.

It periodically queries the Scaleout API to get the status of all clients and records
each client's status in a CSV file. The CSV format makes it easy to import the data
into plotting tools for visualization and analysis of client availability patterns over time.

The script runs continuously, collecting data at regular intervals specified by the user.
Each line in the CSV contains: timestamp, client_name, status.
"""

import time
import csv
import click
import os
from datetime import datetime
from config import settings
from scaleout import Scaleout

@click.command()
@click.option("--csv-filename", "-f", default=None,
              help="CSV filename to store client status data. Defaults to a timestamped filename.")
@click.option("--interval", "-i", default=5,
              help="Time interval in seconds between status checks. Default is 5 seconds.")
def monitor_client_status(csv_filename, interval):
    """Monitor and record the status of Scaleout clients.

    Records one line per client per iteration with timestamp, client_name, and status.
    """
    # Ensure logs directory exists
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Set default filename with timestamp if not provided
    if not csv_filename:
        csv_filename = f"client_status_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    # Prepend logs directory to filename
    csv_path = os.path.join(logs_dir, csv_filename)

    api_client = Scaleout(
        host=settings["DISCOVER_HOST"],
        port=settings["DISCOVER_PORT"],
        secure=settings["SECURE"],
        verify=settings["VERIFY"],
        token=settings["ADMIN_TOKEN"],
    )

    # Create header row
    header = ["timestamp", "client_name", "status"]

    # Create/open CSV file with header if it doesn't exist
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # Check if file is empty
            writer.writerow(header)

    while True:
        fl_clients = api_client.get_clients()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write one row per client
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            for client in fl_clients["result"]:
                client_name = client["name"]
                status = client["status"]
                writer.writerow([timestamp, client_name, status])

        time.sleep(interval)

if __name__ == "__main__":
    monitor_client_status()
