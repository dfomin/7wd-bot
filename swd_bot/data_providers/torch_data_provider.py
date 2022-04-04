import pickle

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
        self.feature_extractor = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        features, cards = self.feature_extractor.features(state)
        features = torch.tensor(features, dtype=torch.float)
        cards = torch.tensor(cards, dtype=torch.float)
        if str(action)[:3] == "Buy":
            action_id = action.card_id
        elif str(action)[:3] == "Dis":
            action_id = action.card_id + EntityManager.cards_count()
        elif str(action)[:3] == "Bui":
            action_id = action.wonder_id + 2 * EntityManager.cards_count()
        else:
            raise ValueError
        winner = state.meta_info["result"].get("winnerIndex", 0)
        return (features, cards), (torch.tensor(action_id, dtype=torch.long), torch.tensor(winner, dtype=torch.long))


class Card2VecDataset(Dataset):
    def __init__(self, states_path: str, actions_path: str):
        with open(states_path, "rb") as f:
            self.states = pickle.load(f)
        with open(actions_path, "rb") as f:
            self.actions = pickle.load(f)
        self.feature_extractor = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        action = self.actions[index]
        action_id = action.card_id + (0 if str(action)[0] == "B" else EntityManager.cards_count())

        state = self.states[index]
        x = StateFeatures.extract_state_features_dict(state)

        c_id = -1
        if action_id in x["available_cards"]:  # берем активную карту
            for i in range(20):
                c = x["cards_board"][i]
                if c == action_id:
                    c_id = i
                    break
        else:  # сбрасываем одну из активных карт
            action_id = action_id - 73
            for i in range(20):
                c = x["cards_board"][i]
                if c == action_id:
                    c_id = i + 20
                    break

        # active_cards = torch.tensor([1 if crd in x["available_cards"] else 0 for crd in x["cards_board"]] * 2,
        #                             dtype=torch.long)
        features, cards = self.feature_extractor.features(state)
        features = torch.tensor(features, dtype=torch.float)
        cards = torch.tensor(cards, dtype=torch.float)
        target = torch.tensor(c_id, dtype=torch.long)

        winner = state.meta_info["result"].get("winnerIndex", 0)

        return (features, cards), (target, winner)


class TorchDataLoader(DataLoader):
    def __init__(self,
                 states_path: str,
                 actions_path: str,
                 batch_size: int,
                 shuffle: bool):
        super().__init__(TorchDataset(states_path, actions_path), batch_size, shuffle)


class Card2VecDataLoader(DataLoader):
    def __init__(self,
                 states_path: str,
                 actions_path: str,
                 batch_size: int,
                 shuffle: bool):
        super().__init__(Card2VecDataset(states_path, actions_path), batch_size, shuffle)
