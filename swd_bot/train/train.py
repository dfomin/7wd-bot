import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim


@hydra.main(config_path="configs", config_name="train")
def train(config: DictConfig):
    model = hydra.utils.instantiate(config["model"])
    train_loader = hydra.utils.instantiate(config["train_data_loader"])
    valid_loader = hydra.utils.instantiate(config["valid_data_loader"])
    epochs = config["epochs"]
    device = torch.device(config["device"])
    output_path = config["output_path"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0
    best_model = None
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        with torch.no_grad():
            correct_pred = 0
            total_pred = 0
            for i, data in enumerate(valid_loader):
                inputs, labels = data
                labels = labels.to(device)

                outputs = model(inputs.to(device))
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions.to(device)):
                    if label == prediction:
                        correct_pred += 1
                    total_pred += 1

        accuracy = round(100 * correct_pred / total_pred)
        print(f'[{epoch + 1}] loss: {running_loss / count:.3f}, Accuracy: {accuracy}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict()
    if best_model is not None:
        torch.save(best_model.state_dict(), f"{output_path}/model_acc{best_accuracy}")

    print('Finished Training')


if __name__ == "__main__":
    train()
