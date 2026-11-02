import pytest
from cv_utils.sync import SyncMatch, filter_ok, match_frames_to_lidar

def test_exact_match():
    matches = match_frames_to_lidar([0.0, 1.0], [0.0, 1.0], max_dt=0.05)
    assert len(matches) == 2
    m0 = matches[0]
    m1 = matches[1]

    assert m0.camera_idx == 0
    assert m0.lidar_idx == 0
    assert m0.dt == pytest.approx(0.0)
    assert m0.quality == 'ok'

    assert m1.camera_idx == 1
    assert m1.lidar_idx == 1
    assert m1.dt == pytest.approx(0.0)
    assert m1.quality == 'ok'    

def test_within_threshold():
    matches = match_frames_to_lidar([0.0], [0.02], max_dt=0.05)
    
    assert len(matches) == 1
    assert matches[0].dt == pytest.approx(0.02)
    assert matches[0].quality == "ok"

def test_above_threshold():
    matches = match_frames_to_lidar([0.0], [0.2], max_dt=0.05)

    assert len(matches) == 1
    assert matches[0].dt == pytest.approx(0.2)
    assert matches[0].quality == "low"

def test_empty_camera():
    matches = match_frames_to_lidar([], [0.0, 0.1], max_dt=0.05)
    
    assert matches == []

def test_empty_lidar_raises():
    """Пустой lidar_ts при непустом camera_ts — ValueError."""
    with pytest.raises(ValueError, match="lidar_ts пуст"):
        match_frames_to_lidar([0.0], [], max_dt=0.05)

def test_nearest_chosen():
    """Из нескольких пакетов выбирается ближайший по времени."""
    # камера 0.5 -> лидары 0.1, 0.48, 0.9 -> ближайший 0.48 (индекс 1)
    matches = match_frames_to_lidar([0.5], [0.1, 0.48, 0.9], max_dt=0.05)
    
    assert len(matches) == 1
    assert matches[0].lidar_idx == 1       # индекс 0.48 в списке
    assert matches[0].dt == pytest.approx(0.02)  # |0.48 - 0.5|
    assert matches[0].quality == "ok"

def test_filter_ok():
    """filter_ok оставляет только quality='ok'."""
    # нужно импортировать SyncMatch и filter_ok в начале файла
    matches = [
        SyncMatch(0, 0, 0.01, "ok"),
        SyncMatch(1, 0, 0.2, "low"),
        SyncMatch(2, 1, 0.03, "ok"),
    ]
    ok = filter_ok(matches)
    
    assert len(ok) == 2
    assert all(m.quality == "ok" for m in ok)

def test_order_preserved():
    """Порядок matches соответствует порядку camera_ts."""
    matches = match_frames_to_lidar(
        [0.0, 0.033, 0.066],   # 3 кадра
        [0.0, 0.1],            # 2 пакета
        max_dt=0.05
    )
    
    assert [m.camera_idx for m in matches] == [0, 1, 2]