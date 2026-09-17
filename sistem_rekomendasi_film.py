#!/usr/bin/env python
# coding: utf-8

# # Sistem Rekomendasi Film — Content-Based Filtering & Collaborative Filtering
# 
# **Nama:** David Kurniawan
# **Kelas:** Dicoding — Belajar Machine Learning Terapan
# **Dataset:** [MovieLens Latest Small (`ml-latest-small`)](https://grouplens.org/datasets/movielens/latest/) — GroupLens Research
# 
# Notebook ini membangun sistem rekomendasi film *top-N* dengan dua pendekatan:
# 
# 1. **Content-Based Filtering (CBF)** — TF-IDF pada genre dan *tag* pengguna, lalu *weighted cosine similarity*.
# 2. **Collaborative Filtering (CF)** — dua algoritma:
#    * **SVD** (*matrix factorization*, library Surprise) dengan *hyperparameter tuning* `GridSearchCV`.
#    * **Neural Collaborative Filtering (NeuMF)** berbasis PyTorch (GPU bila tersedia) yang dilatih dengan *implicit feedback* dan *negative sampling*.
# 
# Urutan notebook sama dengan urutan laporan: **Data Loading → Data Understanding & EDA → Data Preparation → Modeling → Evaluation → Kesimpulan**.

# ## 1. Setup & Import Library
# 
# Sel berikut memasang `scikit-surprise` bila belum ada, mengimpor library, menetapkan *random seed* agar hasil dapat direproduksi, dan mendeteksi perangkat komputasi (CUDA/CPU).

# In[1]:


import importlib.util
import subprocess
import sys

# Pasang scikit-surprise hanya jika belum tersedia (mis. saat dijalankan di Google Colab).
if importlib.util.find_spec("surprise") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-surprise"])


# In[2]:


import copy
import io
import random
import re
import time
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
from IPython.display import display
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV
from torch import nn

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch :", torch.__version__)
print("Device  :", DEVICE, f"({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "")


# In[3]:


# Struktur folder: jika notebook dijalankan dari folder `notebooks/`, root proyek adalah parent-nya.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULT_DIR = PROJECT_ROOT / "outputs" / "results"
for folder in (RAW_DIR, FIG_DIR, RESULT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Konstanta yang dipakai di seluruh notebook
MIN_USER_RATINGS = 20      # user dengan < 20 rating dibuang
MIN_MOVIE_RATINGS = 5      # film dengan < 5 rating dibuang
TEST_SIZE = 0.2            # 80:20 per user
RELEVANCE_THRESHOLD = 4.0  # film dianggap relevan/disukai jika rating >= 4.0
K_VALUES = (5, 10)
TOP_N = 10


def save_fig(name):
    # Simpan figure aktif ke outputs/figures agar dapat dimuat di laporan.
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=120, bbox_inches="tight")
    plt.show()


# ## 2. Data Loading
# 
# Dataset **MovieLens Latest Small** diunduh langsung dari situs GroupLens, kemudian empat berkas CSV (`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`) diekstrak ke `data/raw/`. Jika berkas sudah ada, unduhan dilewati.

# In[4]:


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
FILES = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]

if not all((RAW_DIR / f).exists() for f in FILES):
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for f in FILES:
            (RAW_DIR / f).write_bytes(archive.read(f"ml-latest-small/{f}"))
    print("Dataset berhasil diunduh ke", RAW_DIR)
else:
    print("Dataset sudah tersedia di", RAW_DIR)

movies = pd.read_csv(RAW_DIR / "movies.csv")
ratings = pd.read_csv(RAW_DIR / "ratings.csv")
tags = pd.read_csv(RAW_DIR / "tags.csv")
links = pd.read_csv(RAW_DIR / "links.csv")


# ## 3. Data Understanding
# 
# ### 3.1 Ukuran dan kondisi data
# 
# Langkah pertama adalah melihat jumlah baris/kolom, tipe data, *missing value*, dan duplikat pada setiap tabel.

# In[5]:


tables = {"movies": movies, "ratings": ratings, "tags": tags, "links": links}
summary = pd.DataFrame(
    {
        name: {
            "baris": len(df),
            "kolom": df.shape[1],
            "missing_values": int(df.isna().sum().sum()),
            "duplikat": int(df.duplicated().sum()),
        }
        for name, df in tables.items()
    }
).T
display(summary)

for name, df in tables.items():
    print(f"\n===== {name} =====")
    df.info()


# In[6]:


for name, df in tables.items():
    print(f"\n===== {name}: 5 baris pertama =====")
    display(df.head())


# In[7]:


print("Missing value per kolom:")
for name, df in tables.items():
    print(f"  {name:8s}", df.isna().sum().to_dict())

n_users = ratings["userId"].nunique()
n_rated_movies = ratings["movieId"].nunique()
print(f"\nJumlah film di katalog          : {movies['movieId'].nunique():,}")
print(f"Jumlah user unik                : {n_users:,}")
print(f"Jumlah film yang pernah dirating: {n_rated_movies:,}")
print(f"Jumlah rating                   : {len(ratings):,}")
print(f"Jumlah tag                      : {len(tags):,} (pada {tags['movieId'].nunique():,} film)")
print(f"Sparsity matriks user-item      : {1 - len(ratings) / (n_users * n_rated_movies):.4%}")
print(f"Rentang waktu rating            : {pd.to_datetime(ratings['timestamp'], unit='s').min().date()}"
      f" s.d. {pd.to_datetime(ratings['timestamp'], unit='s').max().date()}")
print(f"Film tanpa genre                : {(movies['genres'] == '(no genres listed)').sum()}")
print(f"Judul film duplikat             : {movies['title'].duplicated().sum()}")
no_year = (~movies["title"].str.contains(r"\(\d{4}\)\s*$")).sum()
print(f"Film tanpa tahun pada judul     : {no_year}")


# In[8]:


display(ratings[["rating"]].describe().T)
per_user = ratings.groupby("userId").size()
per_movie = ratings.groupby("movieId").size()
display(pd.DataFrame({"rating per user": per_user.describe(), "rating per film": per_movie.describe()}).T)


# ### 3.2 Deskripsi variabel
# 
# | Berkas | Variabel | Deskripsi |
# |---|---|---|
# | `movies.csv` | `movieId` | ID unik film (integer). |
# | | `title` | Judul film beserta tahun rilis dalam kurung, mis. `Toy Story (1995)`. |
# | | `genres` | Daftar genre yang dipisahkan `|` (20 label, termasuk `(no genres listed)`). |
# | `ratings.csv` | `userId` | ID unik pengguna (anonim). |
# | | `movieId` | ID film yang dirating (terhubung ke `movies.csv`). |
# | | `rating` | Rating eksplisit skala 0.5–5.0 dengan kelipatan 0.5. |
# | | `timestamp` | Waktu pemberian rating (detik sejak 1 Januari 1970 UTC). |
# | `tags.csv` | `userId` | ID pengguna yang memberi tag. |
# | | `movieId` | ID film yang diberi tag. |
# | | `tag` | Label bebas buatan pengguna, mis. `pixar`, `dark hero`. |
# | | `timestamp` | Waktu pemberian tag. |
# | `links.csv` | `movieId` | ID film MovieLens. |
# | | `imdbId` | ID film di IMDb. |
# | | `tmdbId` | ID film di TMDb (8 nilai kosong). |
# 
# `links.csv` hanya berisi ID eksternal sehingga tidak dipakai dalam pemodelan.

# ### 3.3 Exploratory Data Analysis (EDA)

# In[9]:


counts = ratings["rating"].value_counts().sort_index()
plt.figure(figsize=(8, 4))
sns.barplot(x=counts.index.astype(str), y=counts.values, color="#3b7dd8")
plt.axvline(list(counts.index).index(4.0) - 0.5, color="#d64545", ls="--", label="relevan (≥ 4.0)")
plt.title("Distribusi rating")
plt.xlabel("Rating")
plt.ylabel("Jumlah rating")
plt.legend()
save_fig("01_rating_distribution.png")
print(f"Rata-rata rating: {ratings['rating'].mean():.2f} | median: {ratings['rating'].median()}")
print(f"Proporsi rating >= 4.0: {(ratings['rating'] >= 4).mean():.1%}")


# **Insight:** rating condong ke nilai tinggi — nilai paling sering adalah 4.0, disusul 3.0 dan 5.0, dengan rata-rata ±3.5. Pengguna juga lebih sering memberi angka bulat daripada setengah. Sekitar setengah rating bernilai ≥ 4.0, sehingga ambang **4.0** masuk akal untuk mendefinisikan film yang *relevan/disukai* saat evaluasi.

# In[10]:


genre_counts = movies["genres"].str.split("|").explode().value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette="viridis", legend=False)
plt.title("Jumlah film per genre")
plt.xlabel("Jumlah film")
plt.ylabel("")
save_fig("02_genre_frequency.png")
print("Rata-rata jumlah genre per film:", round(movies["genres"].str.split("|").str.len().mean(), 2))


# **Insight:** *Drama* dan *Comedy* mendominasi katalog, sedangkan *Film-Noir*, *IMAX*, dan `(no genres listed)` sangat jarang. Satu film rata-rata punya lebih dari dua genre. Genre umum seperti Drama kurang membedakan satu film dari film lain, sehingga pembobotan **TF-IDF** (memberi bobot kecil pada genre umum) tepat untuk *content-based filtering*.

# In[11]:


bins = np.logspace(np.log10(per_user.min()), np.log10(per_user.max()), 40)
plt.figure(figsize=(8, 4))
plt.hist(per_user, bins=bins, color="#2a9d8f", edgecolor="white")
plt.xscale("log")
plt.axvline(per_user.median(), color="#d64545", ls="--", label=f"median = {per_user.median():.0f}")
plt.title("Jumlah rating per user")
plt.xlabel("Rating per user (skala log)")
plt.ylabel("Jumlah user")
plt.legend()
save_fig("03_user_activity.png")


# **Insight:** setiap user memiliki minimal 20 rating (sesuai kebijakan GroupLens), tetapi distribusinya sangat miring ke kanan: median sekitar 70 rating, sementara user paling aktif memberi lebih dari 2.600 rating. Pembagian *train/test* dilakukan **per user** agar setiap user tetap terwakili di kedua set.

# In[12]:


sorted_counts = per_movie.sort_values(ascending=False).to_numpy()
cumulative = sorted_counts.cumsum() / sorted_counts.sum()
head = int(np.searchsorted(cumulative, 0.5)) + 1
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(sorted_counts) + 1), sorted_counts, color="#264653")
plt.fill_between(np.arange(1, head + 1), sorted_counts[:head], color="#e9c46a", alpha=0.6,
                 label=f"{head} film teratas = 50% rating")
plt.yscale("log")
plt.title("Popularitas film (long tail)")
plt.xlabel("Peringkat film")
plt.ylabel("Jumlah rating (skala log)")
plt.legend()
save_fig("04_long_tail.png")
print(f"{head} film ({head / len(sorted_counts):.1%} film yang dirating) menerima 50% seluruh rating.")
print(f"Film dengan hanya 1 rating: {(per_movie == 1).sum():,} ({(per_movie == 1).mean():.1%})")
print(f"Film dengan < {MIN_MOVIE_RATINGS} rating: {(per_movie < MIN_MOVIE_RATINGS).sum():,}")


# **Insight:** pola **long tail** terlihat jelas — sebagian kecil film populer menerima separuh rating, sedangkan lebih dari sepertiga film hanya punya satu rating. Film dengan sangat sedikit rating tidak memberi sinyal kolaboratif yang cukup, sehingga film dengan < 5 rating akan difilter pada tahap *data preparation*.

# In[13]:


ratings_dt = pd.to_datetime(ratings["timestamp"], unit="s")
yearly = ratings.groupby(ratings_dt.dt.year)["rating"].agg(["size", "mean"])
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(yearly.index, yearly["size"], color="#8ab17d")
ax.set(title="Jumlah rating per tahun", xlabel="Tahun", ylabel="Jumlah rating")
ax2 = ax.twinx()
ax2.plot(yearly.index, yearly["mean"], color="#d64545", marker="o")
ax2.set_ylabel("Rata-rata rating", color="#d64545")
ax2.grid(False)
save_fig("05_ratings_per_year.png")


# **Insight:** aktivitas rating tidak merata antartahun (ada lonjakan pada beberapa tahun karena segelintir user sangat aktif), dan rata-rata rating per tahun berfluktuasi sekitar 3.3–3.9. Karena pola waktu ini tidak stabil, pembagian data dilakukan secara acak per user, bukan berdasarkan waktu.

# In[14]:


tag_norm = tags["tag"].str.lower().str.strip()
top_tags = tag_norm.value_counts().head(20)
plt.figure(figsize=(8, 6))
sns.barplot(x=top_tags.values, y=top_tags.index, hue=top_tags.index, palette="mako", legend=False)
plt.title("20 tag terpopuler")
plt.xlabel("Frekuensi")
plt.ylabel("")
save_fig("06_top_tags.png")
tagged_share = movies["movieId"].isin(tags["movieId"]).mean()
print(f"Film yang memiliki tag: {tags['movieId'].nunique():,} dari {len(movies):,} ({tagged_share:.1%})")


# **Insight:** tag berisi deskripsi yang kaya makna (*atmospheric*, *superhero*, *thought-provoking*, *disney*, *twist ending*, dll.), tetapi hanya sekitar 16% film yang memiliki tag. Tag terpopuler *in netflix queue* sebenarnya penanda "ingin ditonton" dan bukan deskripsi konten, sehingga menjadi *noise* kecil yang bobotnya diredam oleh IDF. Karena itu genre dan tag akan dibobot secara **terpisah**: film tanpa tag tetap bisa direkomendasikan lewat genre, sedangkan film dengan tag mendapat sinyal tambahan.

# In[15]:


movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "size"]).reset_index()
movie_stats = movie_stats[movie_stats["size"] >= MIN_MOVIE_RATINGS].merge(movies, on="movieId")
exploded = movie_stats.assign(genre=movie_stats["genres"].str.split("|")).explode("genre")
order = exploded.groupby("genre")["mean"].median().sort_values(ascending=False).index
plt.figure(figsize=(9, 6))
sns.boxplot(data=exploded, x="mean", y="genre", order=order, color="#90be6d", fliersize=2)
plt.title(f"Rata-rata rating film per genre (film dengan ≥ {MIN_MOVIE_RATINGS} rating)")
plt.xlabel("Rata-rata rating film")
plt.ylabel("")
save_fig("07_genre_ratings.png")


# **Insight:** *Film-Noir*, *Documentary*, dan *War* memiliki median rating film tertinggi, sedangkan *Horror* paling rendah, disusul *Action*, *Comedy*, dan *Children*. Selisih antargenre tidak besar dibanding sebaran di dalam genre, jadi genre saja tidak cukup untuk memprediksi preferensi — pola kolaboratif antar-user tetap diperlukan.

# In[16]:


top_users = per_user.sort_values(ascending=False).index[:100]
top_movies = per_movie.sort_values(ascending=False).index[:100]
block = (
    ratings[ratings["userId"].isin(top_users) & ratings["movieId"].isin(top_movies)]
    .pivot_table(index="userId", columns="movieId", values="rating")
    .reindex(index=top_users, columns=top_movies)
)
plt.figure(figsize=(6, 6))
plt.imshow(block.notna(), cmap="Greys", aspect="auto", interpolation="nearest")
plt.title("Matriks user-item: 100 user teraktif × 100 film terpopuler\n(sel hitam = ada rating)")
plt.xlabel("Film")
plt.ylabel("User")
plt.grid(False)
save_fig("08_sparsity.png")
print(f"Kepadatan blok terpadat : {block.notna().values.mean():.1%}")
print(f"Kepadatan seluruh matriks: {len(ratings) / (n_users * n_rated_movies):.2%}")


# **Insight:** bahkan pada blok user teraktif × film terpopuler masih banyak sel kosong, dan kepadatan seluruh matriks user-item hanya sekitar 1.7% (**sparsity ± 98.3%**). Kondisi ini adalah alasan utama memakai *matrix factorization* (SVD) dan *embedding* (NeuMF): keduanya mempelajari representasi laten berdimensi rendah yang dapat menggeneralisasi ke pasangan user-film yang belum pernah diamati.

# ## 4. Data Preparation
# 
# Tahapan dilakukan berurutan sebagai berikut (urutan sama dengan laporan):
# 
# 1. Membersihkan data film — mengganti `(no genres listed)` dengan `Unknown`, memecah genre menjadi list, mengekstrak tahun.
# 2. Membersihkan data rating — validasi rentang, hapus duplikat user-film, konversi `timestamp` → `datetime`.
# 3. Membersihkan dan menormalisasi tag, lalu menggabungkannya per film.
# 4. Menyusun fitur konten (token genre + tag) untuk CBF.
# 5. Memfilter *cold-start* user dan film.
# 6. Membagi data *train/test* 80:20 per user.
# 7. *Encoding* `userId`/`movieId` ke indeks berurutan dan membangun matriks user-item.

# ### 4.1 Membersihkan data film

# In[17]:


movies_clean = movies.drop_duplicates(subset="movieId").copy()
movies_clean["title"] = movies_clean["title"].str.strip()
movies_clean["genres"] = movies_clean["genres"].replace("(no genres listed)", "Unknown")
movies_clean["genre_list"] = movies_clean["genres"].str.split("|")
movies_clean["year"] = pd.to_numeric(
    movies_clean["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce"
).astype("Int64")

print("Film dengan genre 'Unknown':", (movies_clean["genres"] == "Unknown").sum())
print("Film tanpa tahun           :", movies_clean["year"].isna().sum())
display(movies_clean.head())


# ### 4.2 Membersihkan data rating

# In[18]:


ratings_clean = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
ratings_clean = ratings_clean[ratings_clean["rating"].between(0.5, 5.0)]
ratings_clean = (
    ratings_clean.sort_values("timestamp")
    .drop_duplicates(subset=["userId", "movieId"], keep="last")
    .sort_values(["userId", "movieId"])
    .reset_index(drop=True)
)
ratings_clean["datetime"] = pd.to_datetime(ratings_clean["timestamp"], unit="s")
print(f"Rating sebelum: {len(ratings):,} | sesudah: {len(ratings_clean):,}")
display(ratings_clean.head())


# ### 4.3 Membersihkan tag dan menggabungkannya per film
# 
# Tag diubah ke huruf kecil, karakter non-alfanumerik diganti spasi (`"Sci-Fi"` → `"sci fi"`), tag kosong dibuang, dan pasangan (film, tag) yang sama dihapus agar satu tag tidak terhitung berkali-kali hanya karena diberikan oleh beberapa user.

# In[19]:


def normalize_text(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


tags_clean = tags.dropna(subset=["tag"]).copy()
tags_clean["tag_clean"] = tags_clean["tag"].map(normalize_text)
tags_clean = tags_clean[tags_clean["tag_clean"] != ""]
tags_clean = tags_clean.drop_duplicates(subset=["movieId", "tag_clean"]).reset_index(drop=True)

tag_docs = tags_clean.groupby("movieId")["tag_clean"].apply(" ".join).rename("tags")
print(f"Tag sebelum: {len(tags):,} | sesudah normalisasi & deduplikasi: {len(tags_clean):,}")
display(tag_docs.head())


# ### 4.4 Menyusun fitur konten untuk Content-Based Filtering
# 
# Setiap genre dijadikan **satu token** (`Sci-Fi` → `scifi`, `Film-Noir` → `filmnoir`) agar tidak terpecah menjadi dua kata. Tag hasil agregasi digabungkan ke tabel film; film tanpa tag diisi string kosong.

# In[20]:


def genre_token(genre):
    return normalize_text(genre).replace(" ", "")


movies_content = movies_clean.merge(tag_docs, on="movieId", how="left")
movies_content["tags"] = movies_content["tags"].fillna("")
movies_content["genre_tokens"] = movies_content["genre_list"].map(
    lambda genres: " ".join(genre_token(g) for g in genres)
)
movies_content["content"] = (movies_content["genre_tokens"] + " " + movies_content["tags"]).str.strip()
display(movies_content[["movieId", "title", "genre_tokens", "tags"]].head())


# ### 4.5 Filter cold-start user dan film
# 
# User dengan < 20 rating dan film dengan < 5 rating dibuang. Filter diulang sampai stabil karena membuang film bisa membuat user turun di bawah ambang (dan sebaliknya).

# In[21]:


def filter_cold_start(df, min_user=MIN_USER_RATINGS, min_movie=MIN_MOVIE_RATINGS, max_iter=10):
    for _ in range(max_iter):
        before = len(df)
        df = df[df["movieId"].map(df["movieId"].value_counts()) >= min_movie]
        df = df[df["userId"].map(df["userId"].value_counts()) >= min_user]
        if len(df) == before:
            break
    return df.reset_index(drop=True)


ratings_filtered = filter_cold_start(ratings_clean)
print(f"Rating : {len(ratings_clean):,} -> {len(ratings_filtered):,}")
print(f"User   : {ratings_clean['userId'].nunique():,} -> {ratings_filtered['userId'].nunique():,}")
print(f"Film   : {ratings_clean['movieId'].nunique():,} -> {ratings_filtered['movieId'].nunique():,}")


# ### 4.6 Train/test split 80:20 per user
# 
# Rating setiap user diacak lalu 20% dijadikan data uji. Minimal satu rating tiap user selalu tetap di data latih sehingga semua user di data uji punya representasi di model. Film di data uji yang tidak pernah muncul di data latih dibuang, karena model kolaboratif tidak dapat menilainya.

# In[22]:


def split_by_user(df, test_size=TEST_SIZE, seed=SEED):
    rng = np.random.default_rng(seed)
    shuffled = df.iloc[rng.permutation(len(df))]
    rank = shuffled.groupby("userId").cumcount()
    n_user = shuffled["userId"].map(shuffled["userId"].value_counts())
    is_test = rank < np.floor(n_user * test_size).clip(upper=n_user - 1)
    return (shuffled[~is_test].sort_index().reset_index(drop=True),
            shuffled[is_test].sort_index().reset_index(drop=True))


train_df, test_df = split_by_user(ratings_filtered)
test_df = test_df[test_df["movieId"].isin(train_df["movieId"])].reset_index(drop=True)
print(f"Train: {len(train_df):,} rating | Test: {len(test_df):,} rating")
print(f"User di train: {train_df['userId'].nunique()} | di test: {test_df['userId'].nunique()}")
print(f"Rating relevan (>= {RELEVANCE_THRESHOLD}) di test: {(test_df['rating'] >= RELEVANCE_THRESHOLD).sum():,}")


# ### 4.7 Encoding ID dan matriks user-item
# 
# `userId` dan `movieId` dipetakan ke indeks 0..n-1 agar bisa dipakai sebagai indeks *embedding* PyTorch dan baris/kolom matriks. Himpunan film di data latih menjadi **kandidat rekomendasi** yang sama untuk semua model sehingga perbandingan adil.

# In[23]:


user_ids = np.sort(train_df["userId"].unique())
item_ids = np.sort(train_df["movieId"].unique())
user_to_idx = {u: i for i, u in enumerate(user_ids)}
item_to_idx = {m: i for i, m in enumerate(item_ids)}
n_users, n_items = len(user_ids), len(item_ids)

train_u = train_df["userId"].map(user_to_idx).to_numpy()
train_i = train_df["movieId"].map(item_to_idx).to_numpy()
user_item = sparse.csr_matrix(
    (train_df["rating"].to_numpy(dtype=np.float32), (train_u, train_i)), shape=(n_users, n_items)
)
print(f"Matriks user-item (train): {user_item.shape} | terisi: {user_item.nnz:,} | "
      f"sparsity: {1 - user_item.nnz / (n_users * n_items):.2%}")


# ## 5. Modeling
# 
# Semua model menghasilkan **matriks skor `[n_users × n_items]`**. Rekomendasi *top-N* untuk seorang user adalah N film dengan skor tertinggi yang **belum pernah ia rating** di data latih. Fungsi bantu berikut dipakai bersama oleh semua model.

# In[24]:


def top_n_from_scores(user_id, score_row, n=TOP_N, score_name="score"):
    scores = score_row.astype(np.float64).copy()
    seen = train_df.loc[train_df["userId"] == user_id, "movieId"].map(item_to_idx)
    scores[seen.to_numpy()] = -np.inf
    top = np.argsort(-scores, kind="stable")[:n]
    out = pd.DataFrame({"movieId": item_ids[top], score_name: scores[top].round(4)})
    return out.merge(movies_clean[["movieId", "title", "genres"]], on="movieId")[
        ["movieId", "title", "genres", score_name]
    ]


def user_history(user_id, n=10):
    hist = train_df[train_df["userId"] == user_id].sort_values(["rating", "timestamp"], ascending=False)
    return hist.head(n).merge(movies_clean[["movieId", "title", "genres"]], on="movieId")[
        ["title", "genres", "rating"]
    ]


SAMPLE_USER = int(user_ids[0])
print(f"Contoh user: {SAMPLE_USER} — 10 film dengan rating tertinggi di data latih:")
display(user_history(SAMPLE_USER))


# ### 5.1 Content-Based Filtering — TF-IDF + weighted cosine similarity
# 
# **Cara kerja**
# 
# 1. **TF-IDF genre** — 20 token genre divektorisasi dengan TF-IDF sehingga genre langka (mis. `filmnoir`) berbobot lebih besar daripada genre umum (`drama`).
# 2. **TF-IDF tag** — tag divektorisasi dengan unigram + bigram (`min_df=2`, maksimal 5.000 fitur).
# 3. Kedua blok dinormalisasi L2, dikalikan `√w` lalu digabung. Hasil kali titik dua baris sama dengan
#    $\text{sim}(a,b) = w \cdot \cos(\text{genre}_a,\text{genre}_b) + (1-w)\cdot\cos(\text{tag}_a,\text{tag}_b)$ dengan $w = 0.5$.
# 
# Genre dan tag dipisah karena jika digabung dalam satu dokumen, film yang tagnya banyak akan "menenggelamkan" genre, sehingga *Toy Story 2* malah kalah mirip dibanding film yang hanya kebetulan bergenre sama.

# In[25]:


TOKEN_PATTERN = r"(?u)\b\w+\b"
GENRE_WEIGHT = 0.5

genre_vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN)
genre_tfidf = genre_vectorizer.fit_transform(movies_content["genre_tokens"])

tag_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, token_pattern=TOKEN_PATTERN)
tag_tfidf = tag_vectorizer.fit_transform(movies_content["tags"])

content_matrix = sparse.hstack(
    [np.sqrt(GENRE_WEIGHT) * normalize(genre_tfidf), np.sqrt(1 - GENRE_WEIGHT) * normalize(tag_tfidf)]
).tocsr()

print("Ukuran TF-IDF genre :", genre_tfidf.shape)
print("Ukuran TF-IDF tag   :", tag_tfidf.shape)
print("Matriks konten      :", content_matrix.shape)
display(
    pd.DataFrame(genre_tfidf[:5].toarray(), columns=genre_vectorizer.get_feature_names_out(),
                 index=movies_content["title"][:5]).round(2)
)


# In[26]:


# Contoh matriks similarity untuk beberapa film
sample_titles = ["Toy Story (1995)", "Toy Story 2 (1999)", "Monsters, Inc. (2001)",
                 "Matrix, The (1999)", "Terminator 2: Judgment Day (1991)", "Godfather, The (1972)"]
rows = [movies_content.index[movies_content["title"] == t][0] for t in sample_titles]
sim_sample = pd.DataFrame(linear_kernel(content_matrix[rows]), index=sample_titles, columns=sample_titles)
plt.figure(figsize=(7, 5))
sns.heatmap(sim_sample, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1)
plt.title("Weighted cosine similarity antar film")
save_fig("09_cbf_similarity_heatmap.png")


# **Catatan:** *Toy Story* dan *Toy Story 2* bergenre identik **dan** berbagi tag `pixar`, sehingga skornya (0.63) lebih tinggi daripada *Monsters, Inc.* yang hanya berbagi genre (0.50). *Monsters, Inc.* tidak memiliki tag, jadi skor maksimumnya 0.5 — bahkan terhadap dirinya sendiri — karena komponen tag bernilai nol. Hal ini tidak mengubah urutan rekomendasi, karena semua kandidat dibandingkan terhadap film acuan yang sama.

# In[27]:


movie_popularity = movies_content["movieId"].map(ratings_clean["movieId"].value_counts()).fillna(0).to_numpy()
title_to_row = pd.Series(movies_content.index, index=movies_content["title"].str.lower())
title_to_row = title_to_row[~title_to_row.index.duplicated()]


def recommend_similar_movies(title, top_n=TOP_N):
    # Top-N film paling mirip dengan `title`; skor sama diurutkan berdasarkan popularitas.
    key = title.strip().lower()
    if key not in title_to_row.index:
        matches = movies_content[movies_content["title"].str.lower().str.contains(key, regex=False)]
        if matches.empty:
            raise ValueError(f"Film '{title}' tidak ditemukan.")
        key = matches.iloc[0]["title"].lower()
    row = title_to_row[key]
    scores = linear_kernel(content_matrix[row], content_matrix).ravel()
    scores[row] = -np.inf
    top = np.lexsort((-movie_popularity, -np.round(scores, 6)))[:top_n]
    result = movies_content.loc[top, ["movieId", "title", "genres"]].copy()
    result["similarity"] = scores[top].round(4)
    return result.reset_index(drop=True)


for query in ["Toy Story (1995)", "Matrix, The (1999)", "Godfather, The (1972)"]:
    ref = movies_content.loc[title_to_row[query.lower()]]
    print(f"\nTop-{TOP_N} film mirip '{query}'  | genre: {ref['genres']} | tag: {ref['tags'][:80]}")
    display(recommend_similar_movies(query))


# Untuk rekomendasi **personal** (dan evaluasi yang sebanding dengan model CF), setiap user dibuatkan **profil konten**: rata-rata vektor konten film yang telah ia rating, dibobot dengan rating yang dikurangi rata-rata rating user tersebut. Film yang disukai (di atas rata-rata) menarik profil mendekat, film yang tidak disukai mendorongnya menjauh. Film kandidat kemudian diurutkan berdasarkan kemiripan dengan profil.

# In[28]:


movie_row = pd.Series(movies_content.index, index=movies_content["movieId"])
item_content = content_matrix[movie_row.loc[item_ids].to_numpy()]

centred = train_df["rating"] - train_df.groupby("userId")["rating"].transform("mean")
flat_user = centred.groupby(train_df["userId"]).transform(lambda r: (r == 0).all())
centred[flat_user] = 1.0
weights = sparse.csr_matrix((centred.to_numpy(), (train_u, train_i)), shape=(n_users, n_items))

user_profiles = normalize(weights @ item_content)
cbf_scores = np.asarray((user_profiles @ item_content.T).todense(), dtype=np.float32)
print("Matriks skor CBF:", cbf_scores.shape)

print(f"\nTop-{TOP_N} rekomendasi Content-Based untuk user {SAMPLE_USER}:")
display(top_n_from_scores(SAMPLE_USER, cbf_scores[user_to_idx[SAMPLE_USER]], score_name="similarity"))


# ### 5.2 Collaborative Filtering #1 — SVD (Matrix Factorization)
# 
# SVD versi Funk (library **Surprise**) memprediksi rating sebagai
# $\hat r_{ui} = \mu + b_u + b_i + q_i^\top p_u$, dengan $\mu$ rata-rata global, $b_u$/$b_i$ bias user/film, serta $p_u$/$q_i$ vektor faktor laten. Parameter dipelajari dengan *stochastic gradient descent* yang meminimalkan galat kuadrat plus regularisasi L2.
# 
# **Hyperparameter tuning:** `GridSearchCV` 5-fold pada **data latih saja** (data uji tidak disentuh) dengan ruang pencarian `n_factors ∈ {50, 100}`, `n_epochs ∈ {20, 30}`, `lr_all ∈ {0.005, 0.01}`, `reg_all ∈ {0.02, 0.1}` (16 kombinasi × 5 fold).

# In[29]:


reader = Reader(rating_scale=(0.5, 5.0))
train_surprise = Dataset.load_from_df(train_df[["userId", "movieId", "rating"]], reader)

param_grid = {
    "n_factors": [50, 100],
    "n_epochs": [20, 30],
    "lr_all": [0.005, 0.01],
    "reg_all": [0.02, 0.1],
    "random_state": [SEED],
}
start = time.perf_counter()
grid = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5, n_jobs=-1)
grid.fit(train_surprise)
print(f"Grid search selesai dalam {time.perf_counter() - start:.1f} detik")

cv_results = pd.DataFrame(grid.cv_results)
cv_table = cv_results[["param_n_factors", "param_n_epochs", "param_lr_all", "param_reg_all",
                       "mean_test_rmse", "std_test_rmse", "mean_test_mae", "rank_test_rmse"]]
cv_table = cv_table.sort_values("rank_test_rmse").reset_index(drop=True)
cv_table.to_csv(RESULT_DIR / "svd_grid_search.csv", index=False)
display(cv_table.round(4))

best_params = {k: v for k, v in grid.best_params["rmse"].items() if k != "random_state"}
print("Parameter terbaik (RMSE):", best_params, "| CV RMSE =", round(grid.best_score["rmse"], 4))


# In[30]:


plt.figure(figsize=(9, 5))
labels = cv_table.apply(
    lambda r: f"k={int(r.param_n_factors)}, ep={int(r.param_n_epochs)}, lr={r.param_lr_all}, reg={r.param_reg_all}",
    axis=1,
)
plt.barh(labels[::-1], cv_table["mean_test_rmse"][::-1], xerr=cv_table["std_test_rmse"][::-1], color="#3b7dd8")
plt.xlim(cv_table["mean_test_rmse"].min() - 0.02, cv_table["mean_test_rmse"].max() + 0.01)
plt.title("SVD GridSearchCV — rata-rata RMSE 5-fold (lebih kecil lebih baik)")
plt.xlabel("RMSE")
plt.yticks(fontsize=7)
save_fig("10_svd_grid_search.png")


# In[31]:


svd_model = SVD(random_state=SEED, **best_params)
svd_model.fit(train_surprise.build_full_trainset())

# Skor untuk seluruh pasangan user-film dihitung sekaligus dari faktor laten (vektorisasi).
trainset = svd_model.trainset
p_u = np.array([svd_model.pu[trainset.to_inner_uid(u)] for u in user_ids])
b_u = np.array([svd_model.bu[trainset.to_inner_uid(u)] for u in user_ids])
q_i = np.array([svd_model.qi[trainset.to_inner_iid(i)] for i in item_ids])
b_i = np.array([svd_model.bi[trainset.to_inner_iid(i)] for i in item_ids])
svd_scores = (trainset.global_mean + b_u[:, None] + b_i[None, :] + p_u @ q_i.T).astype(np.float32)

# Sanity check: hasil vektorisasi sama dengan svd_model.predict()
u0, i0 = SAMPLE_USER, int(item_ids[5])
assert abs(np.clip(svd_scores[user_to_idx[u0], item_to_idx[i0]], 0.5, 5) - svd_model.predict(u0, i0).est) < 1e-4

svd_top = top_n_from_scores(SAMPLE_USER, svd_scores[user_to_idx[SAMPLE_USER]], score_name="predicted_rating")
svd_top["predicted_rating"] = svd_top["predicted_rating"].clip(0.5, 5.0)
print(f"Top-{TOP_N} rekomendasi SVD untuk user {SAMPLE_USER}:")
display(svd_top)


# ### 5.3 Collaborative Filtering #2 — Neural Collaborative Filtering (NeuMF)
# 
# **NeuMF** (He et al., 2017) menggabungkan dua cabang:
# 
# * **GMF** (*Generalized Matrix Factorization*) — perkalian elemen demi elemen *embedding* user dan film (generalisasi MF).
# * **MLP** — *embedding* user dan film (terpisah dari GMF) digabung lalu dilewatkan ke layer `256 → 128 → 64 → 32` (ReLU + Dropout 0.2) untuk menangkap interaksi non-linier.
# 
# Output kedua cabang digabung lalu diproyeksikan ke satu *logit*.
# 
# **Implicit feedback:** tujuan sistem adalah *meranking* film, bukan menebak angka rating. Karena itu setiap film yang pernah dirating user dianggap **positif (1)**, dan untuk setiap positif diambil **4 film acak yang belum pernah dirating** sebagai **negatif (0)**. *Negative sampling* diulang setiap epoch. Model dioptimasi dengan `BCEWithLogitsLoss` dan Adam (lr = 0.001, batch 2048).
# 
# **Early stopping:** 10% rating latih tiap user disisihkan sebagai validasi. Setiap epoch dihitung NDCG@10 validasi, dan bobot epoch terbaik disimpan (*patience* 5, maksimum 30 epoch).

# In[32]:


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, layers=(256, 128, 64, 32), dropout=0.2):
        super().__init__()
        self.user_gmf = nn.Embedding(n_users, emb_dim)
        self.item_gmf = nn.Embedding(n_items, emb_dim)
        self.user_mlp = nn.Embedding(n_users, emb_dim)
        self.item_mlp = nn.Embedding(n_items, emb_dim)
        mlp, in_size = [], emb_dim * 2
        for out_size in layers:
            mlp += [nn.Linear(in_size, out_size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = out_size
        self.mlp = nn.Sequential(*mlp)
        self.output = nn.Linear(in_size + emb_dim, 1)
        for emb in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(self.output.weight, a=1, nonlinearity="sigmoid")
        nn.init.zeros_(self.output.bias)

    def forward(self, users, items):
        gmf = self.user_gmf(users) * self.item_gmf(items)
        mlp = self.mlp(torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=-1))
        return self.output(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)


model_preview = NeuMF(n_users, n_items)
print(model_preview)
print("Jumlah parameter:", f"{sum(p.numel() for p in model_preview.parameters()):,}")


# Fungsi metrik ranking didefinisikan di sini karena dipakai untuk *early stopping* NeuMF dan evaluasi akhir (penjelasan rumus ada di bagian Evaluation).

# In[33]:


def precision_at_k(recommended, relevant, k):
    return sum(1 for item in recommended[:k] if item in relevant) / k


def recall_at_k(recommended, relevant, k):
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    dcg = sum(1 / np.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / idcg


def evaluate_ranking(scores, known_df, target_df, k_values=K_VALUES, threshold=RELEVANCE_THRESHOLD):
    # Film yang sudah dirating di `known_df` disembunyikan; relevan = rating >= threshold di `target_df`.
    masked = scores.astype(np.float64, copy=True)
    masked[known_df["userId"].map(user_to_idx).to_numpy(), known_df["movieId"].map(item_to_idx).to_numpy()] = -np.inf
    liked = target_df[target_df["rating"] >= threshold]
    relevant = {u: set(g) for u, g in liked.groupby("userId")["movieId"]}
    users = sorted(relevant)
    k_max = max(k_values)
    sub = masked[[user_to_idx[u] for u in users]]
    part = np.argpartition(-sub, k_max - 1, axis=1)[:, :k_max]
    top = np.take_along_axis(part, np.argsort(-np.take_along_axis(sub, part, axis=1), axis=1, kind="stable"), axis=1)

    recs = [item_ids[top[r]].tolist() for r in range(len(users))]
    result = {}
    for k in k_values:
        result[f"Precision@{k}"] = np.mean([precision_at_k(recs[r], relevant[u], k) for r, u in enumerate(users)])
        result[f"Recall@{k}"] = np.mean([recall_at_k(recs[r], relevant[u], k) for r, u in enumerate(users)])
        result[f"NDCG@{k}"] = np.mean([ndcg_at_k(recs[r], relevant[u], k) for r, u in enumerate(users)])
    result["evaluated_users"] = len(users)
    result["Coverage@10"] = len(np.unique(top[:, :10])) / n_items
    return result


# In[34]:


@torch.no_grad()
def ncf_score_all(model, batch_users=64):
    model.eval()
    all_items = torch.arange(n_items, device=DEVICE)
    out = np.empty((n_users, n_items), dtype=np.float32)
    for start in range(0, n_users, batch_users):
        users = torch.arange(start, min(start + batch_users, n_users), device=DEVICE)
        logits = model(users.repeat_interleave(n_items), all_items.repeat(len(users)))
        out[start:start + len(users)] = torch.sigmoid(logits).view(len(users), n_items).cpu().numpy()
    return out


def train_neumf(train_part, val_part, max_epochs=30, patience=5, num_negatives=4, batch_size=2048, lr=1e-3):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    users = train_part["userId"].map(user_to_idx).to_numpy()
    items = train_part["movieId"].map(item_to_idx).to_numpy()
    positive_keys = np.unique(users * n_items + items)

    model = NeuMF(n_users, n_items).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history, best_ndcg, best_state, stale = [], -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        start = time.perf_counter()
        # Negative sampling: 4 film acak yang belum pernah dirating untuk setiap interaksi positif
        neg_users = np.repeat(users, num_negatives)
        neg_items = rng.integers(0, n_items, len(neg_users))
        clash = np.isin(neg_users * n_items + neg_items, positive_keys)
        while clash.any():
            neg_items[clash] = rng.integers(0, n_items, clash.sum())
            clash = np.isin(neg_users * n_items + neg_items, positive_keys)

        all_u = torch.as_tensor(np.concatenate([users, neg_users]), device=DEVICE)
        all_i = torch.as_tensor(np.concatenate([items, neg_items]), device=DEVICE)
        labels = torch.cat([torch.ones(len(users)), torch.zeros(len(neg_users))]).to(DEVICE)

        model.train()
        order = torch.randperm(len(labels), device=DEVICE)
        total = 0.0
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(all_u[idx], all_i[idx]), labels[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        val = evaluate_ranking(ncf_score_all(model), train_part, val_part, k_values=(10,))
        history.append({"epoch": epoch, "train_loss": total / len(order),
                        "val_ndcg@10": val["NDCG@10"], "val_recall@10": val["Recall@10"]})
        print(f"Epoch {epoch:02d} | loss {total / len(order):.4f} | val NDCG@10 {val['NDCG@10']:.4f} "
              f"| val Recall@10 {val['Recall@10']:.4f} | {time.perf_counter() - start:.1f}s")

        if val["NDCG@10"] > best_ndcg + 1e-4:
            best_ndcg, best_state, stale = val["NDCG@10"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping — tidak ada perbaikan selama {patience} epoch.")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


ncf_fit_df, ncf_val_df = split_by_user(train_df, test_size=0.1)
start = time.perf_counter()
ncf_model, ncf_history = train_neumf(ncf_fit_df, ncf_val_df)
best_epoch = int(ncf_history.loc[ncf_history["val_ndcg@10"].idxmax(), "epoch"])
print(f"Pelatihan selesai dalam {time.perf_counter() - start:.1f} detik — epoch terbaik: {best_epoch}")


# In[35]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(ncf_history["epoch"], ncf_history["train_loss"], marker="o", color="#264653")
ax1.set(title="Training loss (BCE)", xlabel="Epoch", ylabel="Loss")
ax2.plot(ncf_history["epoch"], ncf_history["val_ndcg@10"], marker="o", label="NDCG@10")
ax2.plot(ncf_history["epoch"], ncf_history["val_recall@10"], marker="o", label="Recall@10")
ax2.axvline(best_epoch, color="#d64545", ls="--", label=f"epoch terbaik = {best_epoch}")
ax2.set(title="Validasi NeuMF", xlabel="Epoch", ylabel="Skor")
ax2.legend()
save_fig("11_ncf_learning_curve.png")


# **Insight:** *loss* latih terus turun, tetapi NDCG@10 validasi berhenti membaik setelah beberapa epoch — tanda model mulai *overfitting* pada interaksi latih. *Early stopping* mengembalikan bobot epoch dengan NDCG@10 validasi tertinggi.

# In[36]:


ncf_scores = ncf_score_all(ncf_model)
print(f"Top-{TOP_N} rekomendasi NeuMF untuk user {SAMPLE_USER}:")
display(top_n_from_scores(SAMPLE_USER, ncf_scores[user_to_idx[SAMPLE_USER]], score_name="score"))


# ## 6. Evaluation
# 
# ### 6.1 Metrik
# 
# **Metrik ranking** (tujuan utama: kualitas daftar *top-N*). Untuk setiap user, film di data uji dengan rating ≥ 4.0 adalah himpunan relevan $Rel$. Film yang sudah dirating di data latih dikeluarkan dari ranking. Nilai akhir adalah rata-rata semua user yang memiliki minimal satu film relevan.
# 
# * $\text{Precision@K} = \dfrac{|Rel \cap Rec@K|}{K}$ — proporsi rekomendasi yang tepat.
# * $\text{Recall@K} = \dfrac{|Rel \cap Rec@K|}{|Rel|}$ — proporsi film relevan yang berhasil ditemukan.
# * $\text{NDCG@K} = \dfrac{DCG@K}{IDCG@K}$, $\;DCG@K = \sum_{i=1}^{K} \dfrac{rel_i}{\log_2(i+1)}$ — memperhitungkan **posisi**: film relevan di peringkat atas bernilai lebih besar. IDCG adalah DCG untuk urutan ideal, sehingga NDCG bernilai 0–1.
# * **Coverage@10** — proporsi katalog kandidat yang muncul minimal sekali di daftar top-10 seluruh user (keragaman).
# 
# **Metrik prediksi rating** (khusus SVD, yang memprediksi angka rating):
# 
# * $\text{RMSE} = \sqrt{\frac{1}{n}\sum (\hat r_i - r_i)^2}$ — menghukum galat besar lebih berat.
# * $\text{MAE} = \frac{1}{n}\sum |\hat r_i - r_i|$ — rata-rata selisih absolut, dalam satuan bintang.
# 
# Sebagai pembanding, dihitung juga **popularity baseline** non-personal: semua user direkomendasikan film dengan jumlah rating ≥ 4.0 terbanyak di data latih.

# In[37]:


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


svd_test_pred = np.array([svd_model.predict(u, i).est for u, i in zip(test_df["userId"], test_df["movieId"])])
global_mean_pred = np.full(len(test_df), train_df["rating"].mean())

rating_table = pd.DataFrame(
    {
        "RMSE": [rmse(test_df["rating"], svd_test_pred), rmse(test_df["rating"], global_mean_pred)],
        "MAE": [mae(test_df["rating"], svd_test_pred), mae(test_df["rating"], global_mean_pred)],
    },
    index=["SVD (tuned)", "Global mean baseline"],
)
display(rating_table.round(4))


# In[38]:


popular_counts = train_df[train_df["rating"] >= RELEVANCE_THRESHOLD]["movieId"].value_counts()
popularity_scores = np.tile(popular_counts.reindex(item_ids, fill_value=0).to_numpy(dtype=np.float32), (n_users, 1))

all_scores = {
    "Content-Based (TF-IDF)": cbf_scores,
    "CF — SVD": svd_scores,
    "CF — NeuMF": ncf_scores,
    "Popularity baseline": popularity_scores,
}
ranking_table = pd.DataFrame({name: evaluate_ranking(s, train_df, test_df) for name, s in all_scores.items()}).T
ranking_table["evaluated_users"] = ranking_table["evaluated_users"].astype(int)
ranking_table.to_csv(RESULT_DIR / "evaluation_metrics.csv")
display(ranking_table.round(4))


# In[39]:


metric_cols = [c for c in ranking_table.columns if c.split("@")[0] in ("Precision", "Recall", "NDCG")]
long = ranking_table[metric_cols].reset_index(names="model").melt(id_vars="model", var_name="metric")
plt.figure(figsize=(12, 5))
ax = sns.barplot(data=long, x="metric", y="value", hue="model", palette="Set2")
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=7, padding=1)
plt.title("Perbandingan metrik ranking pada data uji")
plt.xlabel("")
plt.ylabel("Skor")
plt.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
plt.ylim(0, long["value"].max() * 1.12)
save_fig("12_metric_comparison.png")


# In[40]:


errors = svd_test_pred - test_df["rating"].to_numpy()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(errors, bins=40, kde=True, color="#3b7dd8", ax=ax1)
ax1.axvline(0, color="black", lw=0.8)
ax1.set(title="Distribusi galat prediksi SVD", xlabel="Prediksi − aktual")
by_rating = pd.DataFrame({"rating": test_df["rating"], "abs_err": np.abs(errors)}).groupby("rating")["abs_err"].mean()
ax2.bar(by_rating.index.astype(str), by_rating.values, color="#e76f51")
ax2.set(title="MAE SVD per nilai rating aktual", xlabel="Rating aktual", ylabel="MAE")
save_fig("13_svd_errors.png")


# In[41]:


# Mengapa SVD rendah pada metrik ranking? Bandingkan popularitas film yang direkomendasikan.
train_popularity = train_df["movieId"].value_counts().reindex(item_ids, fill_value=0).to_numpy()


def median_popularity_of_top10(scores):
    masked = scores.astype(np.float64, copy=True)
    masked[train_u, train_i] = -np.inf
    top = np.argsort(-masked, axis=1)[:, :10]
    return float(np.median(train_popularity[top]))


popularity_view = pd.DataFrame(
    {"median jumlah rating latih pada film top-10": {n: median_popularity_of_top10(s) for n, s in all_scores.items()}}
)
popularity_view.loc["(median seluruh kandidat)"] = float(np.median(train_popularity))
display(popularity_view)


# ### 6.2 Analisis hasil
# 
# * **NeuMF adalah model terbaik untuk tujuan top-N.** NeuMF unggul di semua metrik ranking (Precision@10, Recall@10, NDCG@10), termasuk atas *popularity baseline* yang sulit dikalahkan pada MovieLens. Model ini belajar langsung dari pola interaksi yang mirip dengan cara evaluasi dilakukan (film mana yang akan ditonton dan disukai), dan cakupan katalognya jauh lebih luas daripada baseline.
# * **SVD paling baik untuk prediksi rating** — RMSE ±0.84 dan MAE ±0.65, jauh lebih baik dari prediksi rata-rata global. Namun metrik ranking SVD rendah: SVD dioptimasi untuk menebak angka rating pada film yang *sudah* ditonton. Saat meranking ribuan film yang belum ditonton, SVD memilih film yang jarang dirating tetapi berbias tinggi: median jumlah rating film top-10 SVD bahkan di bawah median seluruh kandidat, sedangkan NeuMF memilih film yang cukup dikenal namun tetap personal. Film yang jarang dirating juga jarang muncul di data uji.
# * **Content-Based** lebih baik dari SVD pada ranking dan punya **coverage** tertinggi (rekomendasi beragam), tetapi kalah jauh dari NeuMF karena hanya memakai genre dan tag yang terbatas (±16% film ber-tag). Kelebihannya: tetap bisa merekomendasikan film baru tanpa rating dan hasil *item-to-item*-nya mudah dijelaskan (contoh *Toy Story* → *A Bug's Life*, *Toy Story 2*).
# * Galat SVD terbesar ada pada rating ekstrem, terutama rating rendah 0.5–1.5 (MAE > 1.3) dan rating 5.0: model cenderung menarik prediksi ke arah rata-rata. Galat terkecil ada pada rating 3.5–4.0 yang paling sering muncul.

# ## 7. Kesimpulan
# 
# 1. Data MovieLens (100.836 rating, 610 user, 9.742 film) sangat *sparse* (±98% kosong) dan berpola *long tail*. Filter *cold-start* dan pembagian 80:20 per user menghasilkan data latih dan uji yang representatif.
# 2. **Content-Based Filtering** (TF-IDF genre + tag dengan *weighted cosine similarity*) menghasilkan rekomendasi *item-to-item* yang relevan dan mudah dijelaskan, serta dapat menangani film baru.
# 3. **Collaborative Filtering SVD** yang di-*tuning* dengan GridSearchCV memberikan prediksi rating paling akurat (RMSE/MAE terendah).
# 4. **Collaborative Filtering NeuMF** dengan *implicit feedback* dan *negative sampling* memberikan kualitas daftar top-N terbaik (NDCG@10, Precision@10, Recall@10 tertinggi) dan dipilih sebagai model utama untuk fitur "rekomendasi untuk Anda".
# 5. Pengembangan selanjutnya: *hybrid* (CBF untuk film/user baru + NeuMF untuk user lama), penambahan metadata (sinopsis, pemeran) untuk memperkaya konten, dan evaluasi berbasis waktu (*temporal split*).
