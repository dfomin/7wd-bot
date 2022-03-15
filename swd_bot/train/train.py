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

    action_criterion = nn.CrossEntropyLoss()
    winner_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0
    best_model = None
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        for i, data in enumerate(train_loader):
            inputs, (true_actions, true_winners) = data

            optimizer.zero_grad()

            pred_actions, pred_winners = model(inputs.to(device))
            action_loss = action_criterion(pred_actions, true_actions.to(device))
            winner_loss = winner_criterion(pred_winners, true_winners.to(device))
            loss = action_loss + winner_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        with torch.no_grad():
            correct_actions = 0
            correct_winners = 0
            total_pred = 0
            for i, data in enumerate(valid_loader):
                inputs, (true_actions, true_winners) = data

                pred_actions, pred_winners = model(inputs.to(device))
                _, winner_predictions = torch.max(pred_winners, 1)
                _, action_predictions = torch.max(pred_actions, 1)
                for label, prediction in zip(true_actions, action_predictions.to(device)):
                    if label == prediction:
                        correct_actions += 1
                    total_pred += 1

                for label, prediction in zip(true_winners, winner_predictions.to(device)):
                    if label == prediction:
                        correct_winners += 1

        action_accuracy = round(100 * correct_actions / total_pred, 2)
        winner_accuracy = round(100 * correct_winners / total_pred, 2)
        print(f"[{epoch + 1}] loss: {running_loss / count:.3f}, "
              f"actions: {action_accuracy}%, "
              f"winners: {winner_accuracy}%")

        if action_accuracy > best_accuracy:
            best_accuracy = action_accuracy
            best_model = model.state_dict()
    if best_model is not None:
        torch.save(best_model, f"{output_path}/model_acc{best_accuracy}.pth")

    print('Finished Training')


if __name__ == "__main__":
    train()
