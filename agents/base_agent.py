from abc import ABC, abstractmethod


class BaseMetaAgent(ABC):
    """Абстрактный базовый класс для высокоуровневого агента.

    Позволяет легко менять реализацию без правок main.py:
      - IfElseMetaAgent  (Этап 4)
      - DecisionTreeMetaAgent (Этап 5)
      - MCTSMetaAgent (будущее)
    """

    @abstractmethod
    def choose_path(self, game) -> int:
        """Выбрать следующий узел на карте. Возвращает индекс узла."""
        ...

    @abstractmethod
    def choose_card(self, game) -> int:
        """Выбрать карту из наград. -1 = пропустить."""
        ...

    @abstractmethod
    def choose_shop(self, game) -> int:
        """Выбрать действие в магазине. -1 = выйти."""
        ...

    @abstractmethod
    def choose_event(self, game) -> int:
        """Выбрать вариант события. Возвращает индекс."""
        ...

    @abstractmethod
    def act(self, game):
        """Вернуть spirecomm Action для текущего экрана игры."""
        ...
