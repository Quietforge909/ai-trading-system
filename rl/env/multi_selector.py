"""Round-robin full-universe curriculum selector for aligned multi-coin episodes."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from rl.env.coin_history_context import add_coin_history_context


class MultiCoinSelector:
    def __init__(
        self,
        data_dir: Path,
        start: str,
        end: str,
        episode_days: int,
        universe_size: int,
        seed: int = 42,
        coin_history_context: bool = False,
    ):
        self.hours = episode_days * 24
        self.last_episode_days = int(episode_days)
        self.universe_size = universe_size
        self.coin_history_context = bool(coin_history_context)
        self.rng = random.Random(seed)
        start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        self.frames = {}
        self.indexes = {}
        self.windows = []
        for path in sorted(data_dir.glob("*.parquet")):
            frame = pd.read_parquet(path)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            if self.coin_history_context:
                frame = add_coin_history_context(frame)
            frame = frame.loc[
                (frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts)
            ].sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
            if len(frame) < self.hours:
                continue
            coin = path.stem
            self.frames[coin] = frame
            self.indexes[coin] = pd.Index(frame["timestamp"])
            for offset in range(0, len(frame) - self.hours + 1, self.hours):
                window = frame.iloc[offset : offset + self.hours]
                # A fixed row count is not a fixed time horizon when candles
                # are missing.  Never train across a hidden gap or silently
                # fabricate market/features for an absent hour.
                if window["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
                    self.windows.append((window["timestamp"].iloc[0], coin, offset))
        if not self.windows:
            raise RuntimeError("No multi-coin training windows")
        self.rng.shuffle(self.windows)
        self.cursor = 0
        self.cycles_completed = 0

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def eligible_anchor_coins(self) -> set[str]:
        return {coin for _, coin, _ in self.windows}

    @property
    def completed_selector_cycles(self) -> int:
        return self.cycles_completed

    def next_episode(self) -> tuple[list[str], dict[str, pd.DataFrame]]:
        if self.cursor >= len(self.windows):
            self.cursor = 0
            self.rng.shuffle(self.windows)
            self.cycles_completed += 1
        _, anchor, offset = self.windows[self.cursor]
        self.cursor += 1
        return self._episode_from_window(anchor, offset)

    def _episode_from_window(self, anchor: str, offset: int) -> tuple[list[str], dict[str, pd.DataFrame]]:
        anchor_frame = self.frames[anchor].iloc[offset : offset + self.hours].reset_index(drop=True)
        timeline = pd.Index(anchor_frame["timestamp"])
        selected = [anchor]
        if self.universe_size == 1:
            return selected, {anchor: anchor_frame}
        candidates = [coin for coin in self.frames if coin != anchor]
        self.rng.shuffle(candidates)
        aligned = {anchor: anchor_frame}
        for coin in candidates:
            locations = self.indexes[coin].get_indexer(timeline)
            if (locations < 0).any():
                continue
            aligned[coin] = self.frames[coin].iloc[locations].reset_index(drop=True)
            selected.append(coin)
            if self.universe_size > 0 and len(selected) >= self.universe_size:
                break
        return selected, aligned

    def get_state(self) -> dict:
        return {
            "cursor": self.cursor, "windows": self.windows,
            "cycles_completed": self.cycles_completed,
            "random_state": self.rng.getstate(),
        }

    def load_state(self, state: dict) -> None:
        if state.get("windows") is not None:
            if len(state["windows"]) != len(self.windows):
                raise RuntimeError("Selector window count changed")
            self.windows = state["windows"]
        self.cursor = min(max(0, int(state.get("cursor", 0))), len(self.windows))
        self.cycles_completed = max(0, int(state.get("cycles_completed", 0)))
        if state.get("random_state") is not None:
            self.rng.setstate(state["random_state"])


class YearBalancedCoinSelector(MultiCoinSelector):
    """Full-universe selector that interleaves calendar years without pruning.

    The ordinary selector samples its complete window pool uniformly, which can
    heavily weight recent years when more coins exist there.  This variant
    keeps every timestamp-valid window but chooses the next anchor year
    uniformly, cycling and reshuffling each year's complete window list
    independently.  Calendar year is curriculum scheduling metadata and is
    never included in the policy observation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        grouped: dict[int, list[tuple[pd.Timestamp, str, int]]] = {}
        for window in self.windows:
            grouped.setdefault(int(window[0].year), []).append(window)
        self.year_windows = {year: windows for year, windows in sorted(grouped.items())}
        for windows in self.year_windows.values():
            self.rng.shuffle(windows)
        self.years = tuple(self.year_windows)
        self.year_order = list(self.years)
        self.rng.shuffle(self.year_order)
        self.year_order_cursor = 0
        self.year_cursors = {year: 0 for year in self.years}
        self.year_cycles = {year: 0 for year in self.years}

    @property
    def completed_selector_cycles(self) -> int:
        return min(self.year_cycles.values()) if self.year_cycles else 0

    def next_episode(self) -> tuple[list[str], dict[str, pd.DataFrame]]:
        if self.year_order_cursor >= len(self.year_order):
            self.year_order_cursor = 0
            self.rng.shuffle(self.year_order)
        year = self.year_order[self.year_order_cursor]
        self.year_order_cursor += 1
        windows = self.year_windows[year]
        cursor = self.year_cursors[year]
        if cursor >= len(windows):
            cursor = 0
            self.rng.shuffle(windows)
            self.year_cycles[year] += 1
        _, anchor, offset = windows[cursor]
        self.year_cursors[year] = cursor + 1
        return self._episode_from_window(anchor, offset)

    def get_state(self) -> dict:
        return {
            "year_windows": self.year_windows,
            "year_order": self.year_order,
            "year_order_cursor": self.year_order_cursor,
            "year_cursors": self.year_cursors,
            "year_cycles": self.year_cycles,
            "random_state": self.rng.getstate(),
        }

    def load_state(self, state: dict) -> None:
        saved = state.get("year_windows")
        if not isinstance(saved, dict) or set(int(year) for year in saved) != set(self.years):
            raise RuntimeError("Year-balanced selector window years changed")
        normalized = {int(year): windows for year, windows in saved.items()}
        if {year: len(windows) for year, windows in normalized.items()} != {
            year: len(windows) for year, windows in self.year_windows.items()
        }:
            raise RuntimeError("Year-balanced selector window counts changed")
        self.year_windows = normalized
        order = [int(year) for year in state.get("year_order", self.years)]
        if sorted(order) != list(self.years):
            raise RuntimeError("Year-balanced selector schedule is invalid")
        self.year_order = order
        self.year_order_cursor = min(max(0, int(state.get("year_order_cursor", 0))), len(order))
        self.year_cursors = {
            year: min(max(0, int(state.get("year_cursors", {}).get(year, 0))), len(self.year_windows[year]))
            for year in self.years
        }
        self.year_cycles = {
            year: max(0, int(state.get("year_cycles", {}).get(year, 0)))
            for year in self.years
        }
        if state.get("random_state") is not None:
            self.rng.setstate(state["random_state"])


class MixedHorizonCoinSelector:
    """Deterministic curriculum over several hidden episode horizons.

    Every constituent selector uses all timestamp-valid coins/windows for its
    own horizon. The horizon is never exposed to the policy; only the same
    causal market and portfolio observation is passed to PPO.
    """

    def __init__(
        self, data_dir: Path, start: str, end: str, episode_days: tuple[int, ...],
        universe_size: int, seed: int = 42, coin_history_context: bool = False,
    ):
        horizons = tuple(sorted(set(int(days) for days in episode_days)))
        if not horizons or any(days < 2 for days in horizons):
            raise ValueError("Mixed episode horizons must be at least two days")
        self.horizons = horizons
        self.rng = random.Random(seed)
        self.selectors = {
            days: MultiCoinSelector(
                data_dir, start, end, days, universe_size, seed=seed + index + 1,
                coin_history_context=coin_history_context,
            )
            for index, days in enumerate(horizons)
        }
        self.schedule = list(horizons)
        self.rng.shuffle(self.schedule)
        self.schedule_cursor = 0
        self.last_episode_days = horizons[0]

    @property
    def window_count(self) -> int:
        return sum(selector.window_count for selector in self.selectors.values())

    @property
    def eligible_anchor_coins(self) -> set[str]:
        return set().union(*(selector.eligible_anchor_coins for selector in self.selectors.values()))

    @property
    def completed_selector_cycles(self) -> int:
        # A mixed curriculum covers every configured horizon only after its
        # least-visited constituent selector completes a full cycle.
        return min(
            selector.completed_selector_cycles for selector in self.selectors.values()
        )

    def next_episode(self) -> tuple[list[str], dict[str, pd.DataFrame]]:
        if self.schedule_cursor >= len(self.schedule):
            self.schedule_cursor = 0
            self.rng.shuffle(self.schedule)
        days = self.schedule[self.schedule_cursor]
        self.schedule_cursor += 1
        self.last_episode_days = days
        return self.selectors[days].next_episode()

    def get_state(self) -> dict:
        return {
            "horizons": self.horizons,
            "schedule": self.schedule,
            "schedule_cursor": self.schedule_cursor,
            "last_episode_days": self.last_episode_days,
            "random_state": self.rng.getstate(),
            "selectors": {
                str(days): selector.get_state()
                for days, selector in self.selectors.items()
            },
        }

    def load_state(self, state: dict) -> None:
        if tuple(state.get("horizons", ())) != self.horizons:
            raise RuntimeError("Mixed-horizon selector schedule changed")
        saved = state.get("selectors", {})
        for days, selector in self.selectors.items():
            if str(days) not in saved:
                raise RuntimeError("Mixed-horizon selector state is incomplete")
            selector.load_state(saved[str(days)])
        schedule = [int(days) for days in state.get("schedule", self.horizons)]
        if sorted(schedule) != list(self.horizons):
            raise RuntimeError("Mixed-horizon selector schedule is invalid")
        self.schedule = schedule
        self.schedule_cursor = min(max(0, int(state.get("schedule_cursor", 0))), len(schedule))
        self.last_episode_days = int(state.get("last_episode_days", self.horizons[0]))
        if self.last_episode_days not in self.selectors:
            raise RuntimeError("Mixed-horizon selector last horizon is invalid")
        if state.get("random_state") is not None:
            self.rng.setstate(state["random_state"])
