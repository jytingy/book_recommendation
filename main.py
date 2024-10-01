import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dataframe_function import create_df

df_user = pd.DataFrame()
df_user = create_df('user_read.csv', df_user)
df_user = df_user[df_user['Adventure'] == 1] #dataframe of all the books the user has read (they love Adventure)

df_library = pd.DataFrame()
df_library = create_df('goodreads_data.csv', df_library)

all_genres = set(df_user.columns).union(set(df_library.columns))

#reindex so they have same amount of columns
df_user = df_user.reindex(columns=all_genres, fill_value=0)
df_library = df_library.reindex(columns=all_genres, fill_value=0)

user_profile = df_user.mean(axis=0)

book_similarity = cosine_similarity([user_profile], df_library) 
similarity_df = pd.DataFrame(book_similarity.T, index=df_library.index, columns=['Similarity'])
sorted_recs_df = similarity_df.sort_values(by='Similarity', ascending=False)

top_recs_df = sorted_recs_df[sorted_recs_df['Similarity'] > 0.7] #dataframe where books are 70% similar

rec_book_list = list(top_recs_df.index.values.tolist())
user_book_list = list(df_user.index.values.tolist())

for book in rec_book_list:
    if book in user_book_list:
        rec_book_list.remove(book)
print(rec_book_list)
