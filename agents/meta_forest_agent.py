import os
from config import MODELS_DIR
from agents.meta_tree_agent import DecisionTreeMetaAgent

CARD_MODEL_PATH     = os.path.join(MODELS_DIR, "meta_card_slaythedata_rf.pkl")
CAMPFIRE_MODEL_PATH = os.path.join(MODELS_DIR, "meta_campfire_slaythedata_rf.pkl")
PATH_MODEL_PATH     = os.path.join(MODELS_DIR, "meta_path_slaythedata_rf.pkl")
EVENT_MODEL_PATH    = os.path.join(MODELS_DIR, "meta_event_slaythedata_rf.pkl")
SHOP_MODEL_PATH     = os.path.join(MODELS_DIR, "meta_shop_slaythedata_rf.pkl")


class RandomForestMetaAgent(DecisionTreeMetaAgent):
    """Random Forest мета-агент (обучен с sample_weight по победам).

    Поведение идентично DecisionTreeMetaAgent — переопределяет только пути к моделям.
    """

    def __init__(self):
        self.card_model     = self._load(CARD_MODEL_PATH,     "card_rf")
        self.campfire_model = self._load(CAMPFIRE_MODEL_PATH, "campfire_rf")
        self.path_model     = self._load(PATH_MODEL_PATH,     "path_rf")
        self.event_model    = self._load(EVENT_MODEL_PATH,    "event_rf")
        self.shop_model     = self._load(SHOP_MODEL_PATH,     "shop_rf")
