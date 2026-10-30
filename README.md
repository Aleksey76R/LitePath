# LitePath

A lightweight CV-LiDAR pipeline for short-range autonomous navigation.

Учебный проект (2-й курс ПМИ). Легковесный пайплайн компьютерного зрения и обработки лидарных данных для коротких маршрутов (парковка → подъезд, маршрут в парке) на CPU.

## Стек
Python 3.10+, OpenCV, NumPy, PyTorch → ONNX Runtime, CARLA (офлайн-логи), pytest.

## Структура
- `cv_utils/` — утилиты CV: препроцессинг, синхронизация, fast_ops
- `lidar_utils/` — фильтрация, downsampling, transform
- `models/` — своя модель, обучение, экспорт ONNX, квантование
- `pipelines/` — детекция → трекинг → локализация → планирование
- `sim_adapters/` — адаптер CARLA (офлайн-логи)
- `benchmarks/` — замеры FPS/latency/RAM, ablation-study
- `tests/` — unit-тесты
- `docs/` — допущения, теория, API, метрики
- `notebooks/` — эксперименты
- `scripts/` — вспомогательные скрипты

## Статус
В разработке. См. `docs/assumptions.md`.
