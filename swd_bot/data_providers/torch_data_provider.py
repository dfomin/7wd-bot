import pickle

import torch
from omegaconf import DictConfig
from swd.entity_manager import EntityManager
from torch.utils.data import Dataset, DataLoader

from swd_bot.data_providers.feature_extractor import FeatureExtractor


class TorchDataset(Dataset):
    def __init__(self, states_path: str, actions_path: str, feature_extractor: FeatureExtractor):
        with open(states_path, "rb") as f:
            self.states = pickle.load(f)
        with open(actions_path, "rb") as f:
            self.actions = pickle.load(f)
        self.feature_extractor = feature_extractor

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


class TorchDataLoader(DataLoader):
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 states_path: str,
                 actions_path: str,
                 batch_size: int,
                 shuffle: bool):
        super().__init__(TorchDataset(states_path, actions_path, feature_extractor), batch_size, shuffle)


class TorchDataProvider:
    def __init__(self,
                 train: DictConfig,
                 valid: DictConfig,
                 test: DictConfig,
                 feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
        self.train_data_loader = TorchDataLoader(feature_extractor, **train)
        self.valid_data_loader = TorchDataLoader(feature_extractor, **valid)
        self.test_data_loader = TorchDataLoader(feature_extractor, **test)
