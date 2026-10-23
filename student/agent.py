"""Điểm bắt đầu của bài nộp — sinh viên chỉ cần sửa file này.

Baseline bên dưới nhìn trước đúng một nước. Nó minh họa API và cách mô phỏng
nước đi, nhưng cố ý chưa xử lý ô ngẫu nhiên và chưa phải một thuật toán mạnh.
"""

from __future__ import annotations
 
import math
import time
from functools import lru_cache
 
import numpy as np
 
from game2048.observation import Observation
from game2048.rules import move_board
from game2048.types import Move
from game2048.heuristic import evaluate_board, HeuristicWeights
 
W = HeuristicWeights()   # trọng số heuristic của môi trường
LOSS = -1.0e9            # giá trị của thế cờ hết nước đi
 
 
# ---------------------------------------------------------------------------
# Hàm hỗ trợ: bàn cờ numpy (4, 4) <-> khoá bytes để dùng với lru_cache / dict
# ---------------------------------------------------------------------------
 
 
def _to_board(key: bytes) -> np.ndarray:
    return np.frombuffer(key, dtype=np.uint32).reshape(4, 4)
 
 
@lru_cache(maxsize=400_000)
def _afterstate(key: bytes, move: int) -> tuple[bytes, int] | None:
    """Đi một nước bằng rules.move_board. None nếu nước đi không hợp lệ."""
    after, reward, changed = move_board(_to_board(key), Move(move))
    if not changed:
        return None
    return after.tobytes(), reward
 
 
@lru_cache(maxsize=400_000)
def _evaluate(key: bytes) -> float:
    """Điểm heuristic của môi trường (heuristic.evaluate_board)."""
    return evaluate_board(_to_board(key), W)
 
 
class _Timeout(Exception):
    pass
 
 
class Agent:
    name = "ExpectimaxNumpyAgent"
 
    MAX_DEPTH = 12
    CPROB_THRESHOLD = 1e-3
    SAFETY = 0.85
    CHECK_EVERY = 16
    EXPECTED_MOVES = 4500
 
    def __init__(self) -> None:
        self.last_nodes = 0
        self.last_completed_depth = 0
        self._cache: dict[tuple[bytes, int], float] = {}
        self._deadline = math.inf
        self._nodes = 0
 
    def reset(self, seed: int | None = None) -> None:
        self.last_nodes = 0
        self.last_completed_depth = 0
        self._cache = {}
 
    # -- chia thời gian ------------------------------------------------------
 
    def _budget_ms(self, obs: Observation, time_limit_ms: float) -> float:
        budget = float(time_limit_ms)
        if obs.time_remaining_ms is None:
            return budget
        remaining = float(obs.time_remaining_ms)
        budget = min(budget, remaining)
        if obs.total_time_limit_ms is None:
            return budget
        empties = int(np.count_nonzero(obs.board == 0))
        factor = 0.3 if empties >= 8 else (3.0 if empties <= 3 else 1.0)
        moves_left = max(300, self.EXPECTED_MOVES - obs.move_count)
        return max(1.0, min(budget, factor * remaining / moves_left, 0.1 * remaining))
 
    # -- tìm kiếm ----------------------------------------------------------------
 
    def _tick(self) -> None:
        self._nodes += 1
        if self._nodes % self.CHECK_EVERY == 0 and time.perf_counter() > self._deadline:
            raise _Timeout
 
    def _chance(self, after_key: bytes, depth: int, cprob: float) -> float:
        """Kỳ vọng trên các ô mới có thể xuất hiện sau một nước đi."""
        if depth <= 0 or cprob < self.CPROB_THRESHOLD:
            self._tick()
            return _evaluate(after_key)
        cache_key = (after_key, depth)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
 
        after = _to_board(after_key)
        empty_cells = np.argwhere(after == 0)          # toạ độ (hàng, cột) các ô trống
        if len(empty_cells) == 0:
            return _evaluate(after_key)
        p = cprob / len(empty_cells)
        total = 0.0
        for row, column in empty_cells:
            for tile, probability in ((2, 0.9), (4, 0.1)):
                child = after.copy()                    # bản sao ghi được của bàn cờ
                child[row, column] = tile
                total += probability * self._max(child.tobytes(), depth, p * probability)
        value = total / len(empty_cells)
        self._cache[cache_key] = value
        return value
 
    def _max(self, key: bytes, depth: int, cprob: float) -> float:
        """Người chơi chọn nước đi tốt nhất."""
        best = LOSS
        for move in Move:
            result = _afterstate(key, int(move))
            if result is None:
                continue
            after_key, reward = result
            value = W.immediate_reward * reward + self._chance(after_key, depth - 1, cprob)
            if value > best:
                best = value
        self._tick()
        return best
 
    def _search_root(self, key: bytes, legal: list[Move], depth: int) -> Move:
        best_move = legal[0]
        best_score = -math.inf
        for move in legal:
            result = _afterstate(key, int(move))
            if result is None:
                continue
            after_key, reward = result
            score = W.immediate_reward * reward + self._chance(after_key, depth - 1, 1.0)
            if score > best_score:
                best_score, best_move = score, move
        return best_move
 
    # -- giao diện chính ------------------------------------------------------
 
    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        start = time.perf_counter()
        legal = list(obs.legal_moves)
        self.last_nodes = 0
        self.last_completed_depth = 0
        if len(legal) == 1:
            return legal[0]
 
        key = np.ascontiguousarray(obs.board, dtype=np.uint32).tobytes()
        if len(self._cache) > 500_000:
            self._cache = {}
        self._deadline = start + self._budget_ms(obs, time_limit_ms) * self.SAFETY / 1000.0
        self._nodes = 0
        best = legal[0]
 
        for depth in range(1, self.MAX_DEPTH + 1):
            if depth == 1:
                saved, self._deadline = self._deadline, math.inf
            try:
                best = self._search_root(key, legal, depth)
            except _Timeout:
                break
            finally:
                if depth == 1:
                    self._deadline = saved
            self.last_completed_depth = depth
            if time.perf_counter() > self._deadline:
                break
        self.last_nodes = self._nodes
        return best