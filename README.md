# BEEP

Проект для анализа МРТ-снимков головного мозга: веб-приложение показывает оценку наличия опухоли и изображение с наложенной маской сегментации.

## Запуск

```bash
git clone https://github.com/DomiStjls/BEEP.git
cd BEEP
pip install -r requirements.txt
uvicorn app:app --reload
```

Для работы используются две модели: `model/brain_mri_cnn_model1.h5` и `model/best_model6.onnx`. Файл H5 находится в репозитории; ONNX-файла в нём нет. [Ссылка на модель из прежнего README](https://drive.google.com/file/d/1Zw3upDYeAKRPa69VoEtoYNlU--56xAK5/view?usp=sharing) — проверьте, содержит ли она нужный ONNX-файл. В списке зависимостей также могут отсутствовать пакеты, необходимые для загрузки Keras-модели и обработки файлов.

## Структура

- `app.py` — веб-приложение FastAPI.
- `utils.py` и `onnx_model.py` — подготовка изображения и применение моделей.
- `model/` — файлы моделей; `templates/` и `static/` — интерфейс.
- `metrics/` — скрипты для работы с метриками.

## Примеры результата

| Вход | Предсказание |
|---|---|
| ![Входной снимок](https://github.com/user-attachments/assets/61177deb-6228-4b60-a1de-ac98463428fd) | ![Предсказание](https://github.com/user-attachments/assets/d390822c-5d34-4cb3-a8e0-8781a5896f8e) |
| ![Входной снимок](https://github.com/user-attachments/assets/50dcac80-fc8a-47fa-9e0e-0b240eeab722) | ![Предсказание](https://github.com/user-attachments/assets/56d25202-7eb7-478a-a604-5a55c911d2cf) |

В исходном описании приведены метрики F1 = 0.88, IoU = 0.81 и mAP50 = 0.86; условия их расчёта здесь не указаны.
