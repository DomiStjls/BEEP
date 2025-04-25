# BEEP
____
Проект для автоматической сегментации опухолей на МРТ-снимках головного мозга с помощью нейросети на основе архитектуры **U-Net**.  
Доступен как веб-приложение и может быть запущен локально для тестирования (в этом случае надо скачать модель и поместить её в папку `models/`, см. Ссылки) и дообучения.
_____
Быстрый запуск локально:
```
git clone https://github.com/DomiStjls/BEEP.git
cd BEEP
pip install -r requirements.txt 
uvicorn app:app --reload
```

Отлично, README — это лицо проекта, и его важно сделать понятным и аккуратным. Вот шаблон, который ты можешь адаптировать под свой проект (по сегментации опухолей на МРТ, как я понял):

---

### 🔗 Ссылки

- 📂 **Датасет публчиный**: [Kaggle Brain Tumor Segmentation Dataset](https://www.kaggle.com/datasets/your-dataset-link)  
- 📥 **Скачать обученную модель**: [model](https://drive.google.com/file/d/1Zw3upDYeAKRPa69VoEtoYNlU--56xAK5/view?usp=sharing)  
- 🌐 **Публичная версия приложения**: [Наш сайт](https://prodigally-polite-buzzard.cloudpub.ru/) <--- с большой вероятностью не работает, потому что мы не платим за хостинг

---

### 🧠 Что делает этот проект

- Принимает МРТ-снимок как вход
- Сегментирует область опухоли
- Возвращает бинарную маску (0 — фон, 1 — опухоль)
- Показывает результат на изображении (наложение маски)
- Можно использовать как для визуализации, так и для последующего анализа

---

### 🛠 Используемые технологии

- 🐍 Python, PyTorch, PIL
- 📦 numpy, matplotlib, opencv, keras
- 🌐 FastAPI для веб-интерфейса
- 🧠 U-Net архитектура
- 📊 Визуализация результата и маски

---

### 📸 Пример результата

| Вход | Предсказание |
|------|--------------|
| ![image](https://github.com/user-attachments/assets/61177deb-6228-4b60-a1de-ac98463428fd) | ![image](https://github.com/user-attachments/assets/d390822c-5d34-4cb3-a8e0-8781a5896f8e) |
| ![image](https://github.com/user-attachments/assets/50dcac80-fc8a-47fa-9e0e-0b240eeab722) | ![image](https://github.com/user-attachments/assets/56d25202-7eb7-478a-a604-5a55c911d2cf) |

---

### Метрики используемой модели

|F1 score	| IOU	| mAP50 |
|------|------- | ------|
| 0.88 | 0.81 | 0.86 |


