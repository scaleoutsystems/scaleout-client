import argparse

from scaleoututil.utils.model import ScaleoutModel

from data import get_dataloaders
from model import compile_model, model_hyperparams, save_parameters
from settings import BATCHSIZE_VALID, DATASET_PATH, KEYWORDS

from scaleout import Scaleout

HOST = ""  ## INSERT HOST
TOKEN = ""  ## INSERT TOKEN


def init_seedmodel() -> dict:
    """Used to send a seed model to the server. The seed model is normalized with all training data."""
    dataloader_train, _, _ = get_dataloaders(DATASET_PATH, KEYWORDS, 0, 1, BATCHSIZE_VALID, BATCHSIZE_VALID)
    sc_model = compile_model(**model_hyperparams(dataloader_train.dataset))
    seed_path = "seed.npz"
    scaleout_model:ScaleoutModel = save_parameters(sc_model)
    scaleout_model.save_to_file(seed_path)
    print("Seed saved to seed.npz")


def main():
    parser = argparse.ArgumentParser(description="API example")
    parser.add_argument("--init-seed", action="store_true", required=False, help="Use this flag to send a seed model to the server")
    parser.add_argument("--upload-seed", action="store_true", required=False, help="Use this flag to send a seed model to the server")
    parser.add_argument("--start-session", action="store_true", required=False, help="Use this flag to start session")
    args = parser.parse_args()

    if args.init_seed:
        init_seedmodel()
        return

    if HOST == "":
        print("Please insert HOST and TOKEN in scaleout_api.py")
        return
    if TOKEN:
        api_client = Scaleout(host=HOST, secure=True, verify=True, token=TOKEN)
    else:
        api_client = Scaleout(host=HOST)

    if args.upload_seed:
        init_seedmodel()
        response = api_client.set_active_model("seed.npz")
        print(response)
    elif args.start_session:
        # Depending on the computer hosting the clients this round_timeout might need to increase
        response = api_client.start_session(name="Training", round_timeout=1200)
        print(response)
    else:
        print("No flag passed")


if __name__ == "__main__":
    main()
