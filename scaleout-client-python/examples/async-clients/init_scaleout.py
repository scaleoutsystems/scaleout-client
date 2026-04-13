import click
from config import settings
from scaleout import Scaleout


def init_scaleout(seed_path):
    client = Scaleout(
        host=settings["DISCOVER_HOST"],
        port=settings["DISCOVER_PORT"],
        secure=settings["SECURE"],
        verify=settings["VERIFY"],
        token=settings["ADMIN_TOKEN"],
    )

    result = client.set_active_model(seed_path)
    print(result["message"])


if __name__ == "__main__":
    @click.command()
    @click.argument("seed_path", type=str, default="seed.npz")
    def main(seed_path):
        """Initialize Scaleout with a seed model from the specified path."""
        init_scaleout(seed_path)

    main()
