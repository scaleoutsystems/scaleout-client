"""This scripts starts N_CLIENTS clients using the SDK.

If you are running with a local deployment of Scaleout
using docker compose, you need to make sure that clients
are able to resolve the name "combiner" to 127.0.0.1

One way to accomplish this is to edit your /etc/host,
adding the line:

combiner    127.0.0.1

(this requires root previliges)
"""

import threading
import time
import uuid
from io import BytesIO
from multiprocessing import Process

import click
import numpy as np
from scaleoututil.utils.model import ScaleoutModel
from init_seed import compile_model, make_data
from sklearn.metrics import accuracy_score

from config import settings
from scaleout import EdgeClient
from scaleout.client.edge_client import GrpcConnectionOptions

HELPER_MODULE = "numpyhelper"


def get_api_url(host: str, port: int = None, secure: bool = False):
    if secure:
        url = f"https://{host}:{port}" if port else f"https://{host}"
    else:
        url = f"http://{host}:{port}" if port else f"http://{host}"
    if not url.endswith("/"):
        url += "/"
    return url


def load_parameters(model_bytes_io: BytesIO):
    """Load model parameters from a BytesIO object."""
    model_bytes_io.seek(0)  # Ensure we're at the start of the BytesIO object
    a = np.load(model_bytes_io)
    weights = [a[str(i)] for i in range(len(a.files))]
    return weights


def load_model(model_bytes_io: BytesIO):
    parameters = load_parameters(model_bytes_io)

    model = compile_model()
    n = len(parameters) // 2
    model.coefs_ = parameters[:n]
    model.intercepts_ = parameters[n:]

    return model


def on_train(in_model, client_settings):
    print("Running training callback...")
    epochs = settings["N_EPOCHS"]

    # model = load_model(in_model)

    # X_train, y_train, _, _ = make_data()
    # for i in range(epochs):
    #     model.partial_fit(X_train, y_train)

    # # Prepare updated model parameters
    # updated_parameters = model.coefs_ + model.intercepts_
    # out_model = BytesIO()
    # np.savez_compressed(out_model, **{str(i): w for i, w in enumerate(updated_parameters)})
    # out_model.seek(0)

    out_model = in_model

    # Metadata needed for aggregation server side
    training_metadata = {
        "num_examples": 1,
        "training_metadata": {
            "epochs": epochs,
            "batch_size": 1,
            "learning_rate": 0.01,
        },
    }

    metadata = {"training_metadata": training_metadata}

    return out_model, metadata


def on_validate(in_model: ScaleoutModel):
    model = load_model(in_model.get_stream())

    X_train, y_train, X_test, y_test = make_data()

    # JSON schema
    metrics = {"validation_accuracy": accuracy_score(y_test, model.predict(X_test)), "training_accuracy": accuracy_score(y_train, model.predict(X_train))}

    return metrics


def run_client(name="client", client_id=None, no_discovery=False, intermittent=False, online_for=120):
    """Run a Scaleout client with configurable connection options.

    The client can either connect directly to a combiner or use the discovery service.
    It can also run in continuous or intermittent mode.

    In intermittent mode, the client will:
    1. Wait a random delay between 0 and CLIENTS_MAX_DELAY seconds
    2. Connect to the combiner
    3. Stay online for online_for seconds
    4. Disconnect
    5. Repeat this cycle N_CYCLES times

    In continuous mode, the client will:
    1. Connect to the combiner
    2. Stay connected until the program exits

    Args:
        name (str, optional): Name of the client. Defaults to "client".
        client_id (str, optional): Unique ID for the client. If None, a UUID will be generated.
        no_discovery (bool, optional): If True, connect directly to combiner. If False, use discovery service. Defaults to False.
        intermittent (bool, optional): If True, use intermittent connection mode. Defaults to False.
        online_for (int, optional): In intermittent mode, how long to stay connected in seconds. Defaults to 120.

    """
    if client_id is None:
        client_id = str(uuid.uuid4())

    delay = np.random.randint(0, settings["CLIENTS_MAX_DELAY"])
    print(f"Starting {name} in {delay} seconds")
    time.sleep(delay)

    fl_client = EdgeClient(train_callback=on_train, validate_callback=on_validate)
    fl_client.set_name(name)
    fl_client.set_client_id(client_id)

    if no_discovery:
        combiner_config = GrpcConnectionOptions(host=settings["COMBINER_HOST"], port=settings["COMBINER_PORT"])
    else:
        controller_config = {
            "name": fl_client.name,
            "client_id": fl_client.client_id,
            "package": "local",
        }

        url = get_api_url(host=settings["DISCOVER_HOST"], port=settings["DISCOVER_PORT"], secure=settings["SECURE"])
        result, combiner_config = fl_client.connect_to_api(url, settings["CLIENT_TOKEN"], controller_config)

    fl_client.init_grpchandler(config=combiner_config, client_id=fl_client.client_id, token=settings["CLIENT_TOKEN"])

    if intermittent:
        for i in range(settings["N_CYCLES"]):
            if i != 0:
                fl_client.grpc_handler._reconnect_channel()

            threading.Thread(target=fl_client.run, daemon=True).start()
            time.sleep(online_for)
            fl_client.grpc_handler._disconnect_channel()

            # Sample a delay until the client reconnects
            delay = np.random.randint(0, settings["CLIENTS_MAX_DELAY"])
            time.sleep(delay)
    else:
        fl_client.run()


if __name__ == "__main__":

    @click.command()
    @click.option("--name", "-n", default="client", help="Base name for clients (will be appended with number)")
    @click.option("--no-discovery", is_flag=True, help="Connect to combiner without discovery service")
    @click.option("--intermittent", is_flag=True, help="Use intermittent connection/disconnection mode")
    def main(name, no_discovery, intermittent):
        """Launch multiple federated learning clients that run concurrently.

        This script starts N_CLIENTS (from config) client processes that connect to a Scaleout network.
        Use --name to set a base name for clients, --no-discovery to connect directly to a combiner,
        and --intermittent to simulate clients that periodically disconnect and reconnect.
        """
        # We start N_CLIENTS independent client processes
        processes = []
        for i in range(settings["N_CLIENTS"]):
            p = Process(
                target=run_client,
                args=(
                    f"{name}_{i + 1}",
                    str(uuid.uuid4()),
                    no_discovery,
                    intermittent,
                    settings["CLIENTS_ONLINE_FOR_SECONDS"],
                ),
            )
            processes.append(p)

            p.start()

        for p in processes:
            p.join()

    main()
