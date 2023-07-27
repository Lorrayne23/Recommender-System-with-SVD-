import tarfile
import json
import ast
reviews = []
import pandas as pd
from numpy.linalg import svd
import numpy as np

with tarfile.open("lthing_data.tar.gz") as tar:
    with tar.extractfile("lthing_data/reviews.txt") as file:
        for line in file:
            line_str = line.decode("utf-8").strip()  # Converte bytes em string e remove espaços em branco à esquerda/à direita
            if not line_str.startswith("reviews"):
                continue  # Pela linhas que não começão com review

            # Extrai a parte do dicionário da linha e avalie-a
            # Exemplo : {'comment': 'http://www.lonelymountain.net/books/mar08.html#kindlgc ', 'nhelpful': 0, 'unixtime': 1210636800, 'work': '73940', 'flags': [], 'user': 'gwyneira', 'stars': 4.0, 'time': 'May 13, 2008'}
            record = eval(line_str.split("=", 1)[1])

            if any(x not in record for x in ['user', 'work', 'stars']):
                continue
            reviews.append([record['user'], record['work'], record['stars']])

            # Exemplo ['Wedernoch', '9667839', 5.0]
            #print(len(reviews), "records retrieved")
            if len(reviews) == 397979:
                break


reviews = pd.DataFrame(reviews, columns=["user", "work", "stars"])
print(reviews.head())

# Usuários que fizeram review de mais de 50 livros
usercount = reviews[["work","user"]].groupby("user").count()
usercount = usercount[usercount["work"] >= 50]
print(usercount.head())

# Look for the books who reviewed by more than 50 users
workcount = reviews[["work","user"]].groupby("work").count()
workcount = workcount[workcount["user"] >= 50]
print(workcount.head())

# Keep only the popular books and active users
reviews = reviews[reviews["user"].isin(usercount.index) & reviews["work"].isin(workcount.index)]
print(reviews)

reviewmatrix = reviews.pivot(index="user", columns="work", values="stars").fillna(0)

print(reviewmatrix)

matrix = reviewmatrix.values
u, s, vh = svd(matrix, full_matrices=False)


#The code calculates the cosine similarity between the first column
# and each subsequent column in a matrix (vh), and it identifies and prints
# the column index that has the highest cosine similarity with the first column.

# Find the highest similarity
def cosine_similarity(v, u):
    return (v @ u) / (np.linalg.norm(v) * np.linalg.norm(u))


highest_similarity = -np.inf
highest_sim_col = -1
for col in range(1, vh.shape[1]):
    similarity = cosine_similarity(vh[:, 0], vh[:, col])
    if similarity > highest_similarity:
        highest_similarity = similarity
        highest_sim_col = col

print("Column %d (book id %s) is most similar to column 0 (book id %s)" %
      (highest_sim_col, reviewmatrix.columns[col], reviewmatrix.columns[0]))








