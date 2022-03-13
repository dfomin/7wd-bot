import pickle
from typing import Dict, Any

import torch
from swd.entity_manager import EntityManager
from torch.utils.data import Dataset, DataLoader

from swd_bot.state_features import StateFeatures


class TorchDataset(Dataset):
    def __init__(self, states_path: str, actions_path: str):
        with open(states_path, "rb") as f:
            self.states = pickle.load(f)
        with open(actions_path, "rb") as f:
            self.actions = pickle.load(f)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        flat_features = self.flatten_features(StateFeatures.extract_state_features_dict(state))
        features = torch.tensor(flat_features, dtype=torch.float)
        action_id = action.card_id + (0 if str(action)[0] == "B" else EntityManager.cards_count())
        return features, torch.tensor(action_id, dtype=torch.long)

    @staticmethod
    def flatten_features(x: Dict[str, Any]):
        output = [
            x["age"],
            x["current_player"]
        ]
        output.extend(x["tokens"])
        output.append(x["military_pawn"])
        output.extend(x["military_tokens"])
        output.append(x["game_status"])
        for i in range(2):
            output.append(x["players"][i]["coins"])
            output.extend(x["players"][i]["unbuilt_wonders"])
            output.extend(x["players"][i]["bonuses"])
        for card_id in x["cards_board"]:
            ohe = [0] * EntityManager.cards_count()
            if card_id >= 0:
                ohe[card_id] = 1
                # output.extend(EntityManager.card(card_id).bonuses)
            # else:
            # output.extend([0] * len(EntityManager.card(0).bonuses))
            output.extend(ohe)
        return output


class TorchDataLoader(DataLoader):
    def __init__(self, states_path: str, actions_path: str, batch_size: int, shuffle: bool):
        super().__init__(TorchDataset(states_path, actions_path), batch_size, shuffle)
