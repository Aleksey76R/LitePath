"""Синхронизация кадров камеры и пакетов лидара по таймстемпам.

Политика: nearest timestamp. Для каждого кадра камеры берётся ближайший
по времени лидарный пакет. Если разница превышает max_dt, кадр помечается
как low-quality: данные рассинхронизированы, доверять фузии нельзя.

Осознанное упрощение по сравнению с ROS ApproximateTimeSynchronizer:
не решаем задачу глобального оптимального сопоставления, а берём nearest
для каждого кадра независимо. Для низкой скорости (10–15 км/ч) и известного
маршрута этого достаточно.

См. docs/assumptions.md, пункт 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

@dataclass(frozen=True)
class SyncMatch:
    camera_idx: int
    lidar_idx: int
    dt: float
    quality: str

def match_frames_to_lidar(
    camera_ts: Sequence[float],
    lidar_ts: Sequence[float],
    max_dt: float = 0.05,
) -> List[SyncMatch]:
    
    if len(camera_ts) == 0:
        return []

    if len(lidar_ts) == 0:
        raise ValueError("lidar_ts пуст, но camera_ts содержит кадры")

    cam = np.asarray(camera_ts, dtype=np.float64)
    lid = np.asarray(lidar_ts, dtype=np.float64)

    matches: List[SyncMatch] = []

    for i, ts in enumerate(cam):
        j = int(np.argmin(np.abs(lid - ts)))

        dt = float(abs(lid[j] - ts))

        quality = "ok" if dt <= max_dt else "low"

        matches.append(
            SyncMatch(camera_idx=i, lidar_idx=j, dt=dt, quality=quality)
        )

    return matches


def filter_ok(matches: Sequence[SyncMatch]) -> List[SyncMatch]:
    return [m for m in matches if m.quality == "ok"]