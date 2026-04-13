"""This script analyzes client participation in a Scaleout session.

It retrieves data from the Scaleout API about a specific training session (or the most recent one
if not specified) and generates a plot showing the number of aggregated models and validations
per round. This visualization helps in understanding client participation patterns and the
overall health of the federated learning process across training rounds.

The script can be used to monitor client engagement and identify potential issues with
client participation or model validation in the federated learning network.
"""

import click
from config import settings
from scaleout import Scaleout
import matplotlib.pyplot as plt
from datetime import datetime, timezone

def get_latest_session_id(client):
    """Get the most recent session ID from the API."""
    sessions = client.get_sessions()
    if "result" in sessions and sessions["result"]:
        # Sort sessions by committed_at in descending order
        # Parse the date string to ensure proper date comparison

        def parse_date(date_str):
            try:
                return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
            except (ValueError, TypeError):
                # Return a very old date as fallback for invalid dates
                return datetime(1900, 1, 1, tzinfo=timezone.utc)

        sorted_sessions = sorted(
            sessions["result"],
            key=lambda x: parse_date(x.get("committed_at", "")),
            reverse=True
        )
        for session in sorted_sessions:
            print(f"Session: {session['session_id']} - Committed at: {session['committed_at']}")
        return sorted_sessions[0]["session_id"]
    return None

def plot_aggregation_data(session_id):
    """Plot aggregation data for the specified session."""
    client = Scaleout(
        host=settings["DISCOVER_HOST"],
        port=settings["DISCOVER_PORT"],
        secure=settings["SECURE"],
        verify=settings["VERIFY"],
        token=settings["ADMIN_TOKEN"],
    )

    if not session_id:
        session_id = get_latest_session_id(client)
        if not session_id:
            print("No sessions found.")
            return
        print(f"Using latest session: {session_id}")

    rounds = client.get_rounds()
    round_ids = []
    nr_aggregated_models = []
    nr_validations_per_round = []

    for round in rounds["result"]:
        if "combiners" in round and round["round_config"]["session_id"] == session_id:
            round_ids.append(round["round_id"])
            nr_aggregated_models.append(round["combiners"][0]["data"]["aggregation_time"]["nr_aggregated_models"])

            model_id = round["round_config"]["model_id"]
            validations = client.get_validations(model_id=model_id)
            nr_validations_per_round.append(len(validations["result"]))

    print(f"Round IDs: {round_ids}")
    print(f"Number of aggregated models: {nr_aggregated_models}")
    print(f"Number of validations per round: {nr_validations_per_round}")

    # Create the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(round_ids, nr_aggregated_models, marker="o", label="Aggregated Models")
    plt.plot(round_ids, nr_validations_per_round, marker="s", label="Validations")

    # Customize the plot
    plt.xlabel("Round ID")
    plt.ylabel("Count")
    plt.title(f"Session ID: {session_id}")
    plt.legend()
    plt.grid(True)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    @click.command()
    @click.option("--session-id", "-s", default=None, help="Session ID to analyze")
    def main(session_id):
        """Plot the number of aggregated models and validations per round for a session.

        If no session ID is provided, the most recent session will be used.
        """
        plot_aggregation_data(session_id)

    main()
