python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt

Склонировать следующие репозитории: 
git clone https://github.com/nwojke/deep_sort.git
git clone https://github.com/Verg-Avesta/CounTR.git
Изменить:
- в файле /content/CounTR/util/misc.py from torch._six import inf на inf = float('inf')
- в файле /content/CounTR/util/pos_embed.py omega = np.arange(embed_dim // 2, dtype=np.float) на omega = np.arange(embed_dim // 2, dtype=float)

Запустить streamlit:
streamlit run app.py

