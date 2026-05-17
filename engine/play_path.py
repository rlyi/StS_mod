from typing import List

from engine.battle_state import BattleState, Play
from engine.converter import battlestate_deepcopy


class PlayPath:
    def __init__(self, plays: List[Play], state: BattleState):
        self.plays: List[Play] = plays
        self.state: BattleState = state

    def end_turn(self):
        self.state.end_turn()

    def is_valid(self):
        if self.state.must_discard and len(self.plays) == 0:
            return False
        if self.state.amount_to_exhaust > 0:
            return False
        return True


def get_paths_bfs(state: BattleState, max_path_count: int):
    explored_paths: dict = {}
    unexplored_paths = [PlayPath([], battlestate_deepcopy(state))]

    while len(explored_paths) < max_path_count:
        if len(unexplored_paths) == 0:
            break
        path = unexplored_paths.pop()

        path_state = path.state.get_state_hash()
        if path_state in explored_paths:
            continue

        if path.state.amount_to_discard:
            plays = path.state.get_discards()
        elif path.state.amount_to_exhaust:
            plays = path.state.get_exhausts()
        else:
            plays = path.state.get_plays()

        for play in plays:
            new_state: BattleState = battlestate_deepcopy(path.state)
            new_state.transform_from_play(play, is_first_play=not path.plays)
            new_plays: List[Play] = path.plays.copy()
            new_plays.append(play)
            unexplored_paths.append(PlayPath(new_plays, new_state))

        if path.is_valid():
            explored_paths[path_state] = path

    return explored_paths
