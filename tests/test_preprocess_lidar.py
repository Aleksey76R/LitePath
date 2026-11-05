import numpy as np
import pytest

from lidar_utils.preprocess import (
    filter_by_height,
    remove_outliers,
    voxel_downsample,
    preprocess_lidar,
)

def test_filter_by_height_keeps_valid():
    """Точки внутри диапазона остаются."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 2.5],
        [2.0, 2.0, -0.5]
    ])
    filtered = filter_by_height(points, z_min=-1.0, z_max=3.0)
    assert len(filtered) == 3


def test_filter_by_height_removes_outliers():
    """Точки ниже и выше отбрасываются."""
    points = np.array([
        [0.0, 0.0, -1.5],  # ниже
        [1.0, 1.0, 0.0],   # в норме
        [2.0, 2.0, 4.0]    # выше
    ])
    filtered = filter_by_height(points, z_min=-1.0, z_max=3.0)
    assert len(filtered) == 1
    assert filtered[0][2] == 0.0


def test_filter_by_height_empty():
    """Пустой массив возвращается как есть."""
    points = np.empty((0, 3))
    filtered = filter_by_height(points)
    assert len(filtered) == 0


# --- Тесты для remove_outliers ---

def test_remove_outliers_keeps_cluster():
    """Плотное облако остается, шум уходит."""
    cluster = np.random.uniform(0, 0.1, size=(100, 3))
    noise = np.array([[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]])
    points = np.vstack([cluster, noise])
    
    filtered = remove_outliers(points, k=10, std_ratio=2.0)
    assert len(filtered) == 100


def test_remove_outliers_removes_far_point():
    """Одиночная далекая точка удаляется."""
    cluster = np.random.uniform(0, 0.1, size=(50, 3))
    outlier = np.array([[100.0, 100.0, 100.0]])
    points = np.vstack([cluster, outlier])
    
    filtered = remove_outliers(points, k=10, std_ratio=1.0)
    assert len(filtered) == 50
    assert not np.any(np.all(filtered == outlier, axis=1))


def test_remove_outliers_too_few_points():
    """Если точек меньше k, возвращаем без изменений."""
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    filtered = remove_outliers(points, k=10)
    assert len(filtered) == 3

def test_voxel_downsample_reduces_count():
    """100 точек в одном вокселе превращаются в 1."""
    points = np.random.uniform(0, 0.05, size=(100, 3))
    res = voxel_downsample(points, voxel_size=0.15)
    assert len(res) == 1


def test_voxel_downsample_keeps_shape():
    """Форма сохраняется, включая 4-й канал (интенсивность)."""
    points = np.random.uniform(0, 10, size=(50, 4))
    res = voxel_downsample(points, voxel_size=1.0)
    assert res.shape[1] == 4


def test_voxel_downsample_invalid_size():
    """При voxel_size <= 0 выбрасывается ValueError."""
    points = np.random.uniform(0, 10, size=(50, 3))
    with pytest.raises(ValueError, match="voxel_size"):
        voxel_downsample(points, voxel_size=0)

def test_preprocess_lidar_pipeline():
    """Общий тест: 1000 случайных точек проходят весь пайплайн."""
    np.random.seed(42)
    points = np.random.uniform(-2, 5, size=(1000, 3))
    
    result = preprocess_lidar(points, voxel_size=0.5, z_min=-1.0, z_max=3.0)
    
    assert len(result) < 1000
    assert np.all(result[:, 2] >= -1.0)
    assert np.all(result[:, 2] <= 3.0)