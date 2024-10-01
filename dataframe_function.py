import pandas as pd

def create_df(csv_file, df_name):
    df = pd.read_csv(csv_file)
    genre_list = df['Genres'].values.tolist()
    book_list = df['Book'].values.tolist()
    genres = []

    #getting list of genre lists
    for i in genre_list:
        clean_genre_list = []
        for character in i:
            if character.isalpha() or character == ' ':
                clean_genre_list.append(character)
        genre_string = ''.join(clean_genre_list)
        new_list = genre_string.split()
        genres.append(new_list)

    #list of unique genres
    unique_genres = sorted(set(genre for sublist in genres for genre in sublist))

    #binary matrix where rows are books and columns are genres
    binary_matrix = []

    for genres in genres:
        row = [1 if genre in genres else 0 for genre in unique_genres]  
        binary_matrix.append(row)

    df_name = pd.DataFrame(binary_matrix, index=book_list, columns=unique_genres)
    
    return df_name

