"""Модуль предобработки лидарных данных (LiDAR preprocessing).

Этот модуль реализует базовый конвейер очистки сырого облака точек, 
необходимый для обеспечения требований по производительности (FPS ≥ 15) 
и надежности навигации на CPU. 

Сырое облако точек содержит до 80% избыточной информации (земля, небо, шум), 
которая не нужна планировщику траектории. Модуль последовательно выполняет 
три стандартных шага фильтрации:

1. filter_by_height: Отсечение по высоте Z. Убирает землю и объекты 
   выше машины, мгновенно сокращая объем данных на 60-80%.
2. remove_outliers: Статистическое удаление выбросов (SOR). Убирает 
   одиночные шумовые точки (дождь, пыль), чтобы они не воспринимались 
   как ложные препятствия.
3. voxel_downsample: Воксельный даунсемплинг. Прореживает облако в 5-10 раз, 
   сохраняя форму объектов (бордюров, стен), что критично для CPU-инференса.

Функция preprocess_lidar объединяет все три шага в единый пайплайн.

Система координат:
    X — вперёд по направлению движения.
    Y — влево.
    Z — вверх.
Единицы измерения: метры.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

def filter_by_height(points, z_min=-1.0, z_max=3.0):
    if z_min > z_max:
        raise ValueError("z_min не может быть больше z_max")

    if len(points) == 0:
        return points

    mask = (points[:, 2]>=z_min)&(points[:, 2]<=z_max)

    return points[mask]

def remove_outliers(points, k=10, std_ratio=2.0):
    if len(points)==0:
        return points
    if len(points)<k:
        return points

    tree = cKDTree(points)
    distances, _ = tree.query(points, k+1)
    mean_dist = distances[:, 1:].mean(axis=1)
    threshold = mean_dist.mean() + std_ratio * mean_dist.std()

    mask = mean_dist <= threshold
    return points[mask]

def voxel_downsample(points, voxel_size=0.15):
    if voxel_size <= 0:
        raise ValueError("voxel_size не может быть <= 0")
        
    if len(points) == 0:
        return points

    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    
    _, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)
    
    num_dims = points.shape[1]
    result = np.zeros((inverse.max() + 1, num_dims))
    np.add.at(result, inverse, points)
    
    counts = np.bincount(inverse).reshape(-1, 1)
    return result / counts

def preprocess_lidar(points, voxel_size=0.15, z_min=-1.0, z_max=3.0):
    points = filter_by_height(points, z_min, z_max)
    points = remove_outliers(points)
    points = voxel_downsample(points, voxel_size)
    return points