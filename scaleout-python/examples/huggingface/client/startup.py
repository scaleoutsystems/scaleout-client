import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scaleout import EdgeClient, ScaleoutModel
from scaleoututil.helpers.helpers import get_helper

from model import load_parameters, save_parameters, compile_model
from data import load_data, prepare_data


HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

MODEL = "google/bert_uncased_L-2_H-128_A-2"

def startup(client: EdgeClient):
    prepare_data()
    MyClient(client)


class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class MyClient():
    def __init__(self, client: EdgeClient):
        self.client = client
        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)
    

    def preprocess(self, text):
        """Preprocesses text input.

        :param text: The text to preprocess.
        :type text: str
        """
        text = text.lower()
        text = text.replace("\n", " ")
        return text


    def train(self, scaleout_model: ScaleoutModel, settings, data_path=None, batch_size=2, epochs=1, lr=1e-3):
        model = load_parameters(scaleout_model)

        X_train, y_train = load_data(data_path, is_train=True)
        # preprocess
        X_train = [self.preprocess(text) for text in X_train]
        # encode
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        train_encodings = tokenizer(X_train, truncation=True, padding="max_length", max_length=512)
        train_dataset = SpamDataset(train_encodings, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optim = torch.optim.AdamW(model.parameters(), lr=lr)

        criterion = torch.nn.CrossEntropyLoss()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        global_step = 0
        for epoch in range(epochs):
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)

                loss = criterion(outputs.logits, labels)
                loss.backward()
                optim.step()

                global_step += 1
                stats = {"loss": loss.item()}
                if global_step % 100 == 0:
                    self.client.log_metric(stats, step=global_step)

        # Metadata needed for aggregation server side
        metadata = {
            # num_examples are mandatory
            "num_examples": len(train_dataset),
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
        }

        self.client.check_task_abort()

        result_model = save_parameters(model)

        return result_model, {"training_metadata": metadata}


    def validate(self, scaleout_model: ScaleoutModel, data_path=None):
        """Validate model.

        :param scaleout_model: The ScaleoutModel object containing the model.
        :type scaleout_model: ScaleoutModel
        :param data_path: The path to the data file.
        :type data_path: str
        """
        # Load data
        X_train, y_train = load_data(data_path, is_train=True)
        X_test, y_test = load_data(data_path, is_train=False)

        # preprocess
        X_test = [self.preprocess(text) for text in X_test]
        X_train = [self.preprocess(text) for text in X_train]

        # test dataset
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        test_encodings = tokenizer(X_test, truncation=True, padding="max_length", max_length=512)
        test_dataset = SpamDataset(test_encodings, y_test)

        # Load model
        model = load_parameters(scaleout_model)
        model.eval()

        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()

        # test set validation
        with torch.no_grad():
            correct = 0
            total_loss = 0
            total = 0
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, dim=1)  # index of the max logit

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item() * labels.size(0)

        test_accuracy = correct / total
        print(f"Accuracy: {test_accuracy * 100:.2f}%")

        test_loss = total_loss / total
        print("test loss: ", test_loss)

        # Return metrics
        metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }

        return metrics