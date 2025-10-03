# Практична №3 – Рекомендаційна система без surprise

import pandas as pd
import numpy as np

# -------------------------------
# 1. Завантаження та підготовка даних
# -------------------------------

url_data = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
columns_data = ['user_id', 'movie_id', 'rating', 'timestamp']
data = pd.read_csv(url_data, sep='\t', names=columns_data)

print("Перші 5 рядків даних:")
print(data.head(), "\n")

# Перевірка пропусків
print("Пропуски у даних:")
print(data.isnull().sum(), "\n")

# Створюємо матрицю користувач-об’єкт
user_item_matrix = data.pivot(index='user_id', columns='movie_id', values='rating')
print("Матриця користувач-об’єкт:")
print(user_item_matrix.head(), "\n")

# -------------------------------
# 2. Рекомендації на основі популярності
# -------------------------------
popular_movies = data.groupby('movie_id')['rating'].mean().sort_values(ascending=False)
print("Топ-10 найпопулярніших фільмів (за середньою оцінкою):")
print(popular_movies.head(10), "\n")

# -------------------------------
# 3. Рекомендації на основі схожості користувачів (User-Based)
# -------------------------------

# Заповнюємо пропуски середнім рейтингом по фільму
matrix_filled = user_item_matrix.fillna(user_item_matrix.mean(axis=0))

# Розрахунок схожості користувачів (кореляція Пірсона)
user_similarity = matrix_filled.T.corr()
print("Схожість між користувачами:")
print(user_similarity.head(), "\n")

# Функція для прогнозу рейтингу на основі схожих користувачів
def predict_rating(user_id, movie_id):
    # Користувачі, які оцінили цей фільм
    other_users = matrix_filled[movie_id].dropna()
    # Схожість з цим користувачем
    sims = user_similarity[user_id].loc[other_users.index]
    # Вага середнього рейтингу
    if sims.sum() == 0:
        return matrix_filled[movie_id].mean()
    return np.dot(sims, other_users) / sims.sum()

# -------------------------------
# 4. Генерація ТОП-5 рекомендацій для користувача
# -------------------------------
user_id = 1
all_movie_ids = data['movie_id'].unique()
rated_movies = data[data['user_id'] == user_id]['movie_id'].tolist()
unrated_movies = [m for m in all_movie_ids if m not in rated_movies]

predictions = [(m, predict_rating(user_id, m)) for m in unrated_movies]
predictions.sort(key=lambda x: x[1], reverse=True)

top5 = predictions[:5]
print(f"ТОП-5 рекомендацій для користувача {user_id}:")
for movie_id, rating in top5:
    print(f"Фільм ID {movie_id} - прогнозована оцінка: {rating:.2f}")

# -------------------------------
# 5. Інтерпретація результатів
# -------------------------------
print("\nІнтерпретація:")
print("- Сильні сторони: персоналізовані рекомендації на основі схожості користувачів.")
print("- Слабкі сторони: не враховуються жанри та атрибути фільмів, може бути повільно на великих наборах даних.")

