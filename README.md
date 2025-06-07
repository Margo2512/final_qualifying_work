python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt

Склонировать следующие репозитории:

git clone https://github.com/nwojke/deep_sort.git

git clone https://github.com/Verg-Avesta/CounTR.git

Изменить:
- в файле /content/CounTR/util/misc.py from torch._six import inf на inf = float('inf')
- в файле /content/CounTR/util/pos_embed.py omega = np.arange(embed_dim // 2, dtype=np.float) на omega = np.arange(embed_dim // 2, dtype=float)

Скачайте веса и добавьте в папку model: 

https://drive.google.com/file/d/1CzYyiYqLshMdqJ9ZPFJyIzXBa7uFUIYZ/view?usp=sharing

Запуск основного сервиса: streamlit run app.py

Запуск тестов: PYTHONPATH=путь_к_проекту pytest tests/test_metrics.py -v

Конвертация из YOLO формата в CVAT аннотацию: fromYOLOtoCVAT.py

Подсчет контролируемых метрик и визуализация сопоставлений истинных ограничивающих рамок с предсказанными: metrics.ipynb

