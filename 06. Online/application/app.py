# Keep in memory test 
# https://discuss.streamlit.io/t/how-to-add-records-to-a-dataframe-using-python-and-streamlit/19164/6
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#            import            #
#______________________________#

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer, util
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import( OneHotEncoder, StandardScaler, LabelEncoder )
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#         definitions          #
#______________________________#

def preprocessorPipeline(X):
    numeric_features = X.select_dtypes([np.number]).columns 
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    categorical_features = X.select_dtypes("object").columns
    categorical_transformer = Pipeline(
        steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first'))
        ])
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor



def test_set_from(film, target_name, dataset_columns):
    my_film = film
    my_film["genres"] = my_film['genre']

    modelST = SentenceTransformer('all-MiniLM-L6-v2')

    my_film['genre'] = ''
    my_film['themaScore'] = 0.0

    for i in range( len(my_film) ):
        resume = my_film.iloc[i]['resume']
        genres = my_film.iloc[i]['genres']
        bestScore = 0.0
        bestGenre = ''
        for genre in genres:
            emb1 = modelST.encode(str(genre))
            emb2 = modelST.encode(str(resume))
            cos_sim = util.cos_sim(emb1, emb2)
            result = cos_sim.tolist()[0][0]
            if result > bestScore:
                bestScore = result
                bestGenre = genre
        # print(i, '/', len(dataset))
        my_film.themaScore[i] = bestScore
        my_film.genre[i] = bestGenre

    my_film = my_film.drop(columns=['genres','title','resume'])
    my_film = my_film.reindex(columns = [col for col in my_film.columns if col != target_name] + [target_name])
    my_film = my_film[dataset_columns]

    X_test = my_film.drop(columns= [target_name])
    Y_test = my_film[:][target_name]

    return X_test, Y_test, my_film



@st.cache(persist=True, allow_output_mutation=True)
def load_model():
    model = joblib.load("src/random_forest.joblib")
    return model



@st.cache(persist=True, allow_output_mutation=True)
def load_data(target_name):
    # Import Dataset
    dataset = pd.read_csv('src/dataset.csv', delimiter=',', on_bad_lines='skip')
    # dataset = dataset.drop(columns=['tconst', 'numVotes'])
    #
    # Creation columns pour trier dans le bon ordre l'input
    dataset_columns = dataset.columns.to_list()
    #
    # Separation Varibles / Target
    X = dataset.drop(columns= [target_name])

    preprocessor = preprocessorPipeline(X)

    X_train = dataset.drop(columns= [target_name])
    Y_train = dataset[:][target_name]

    return dataset_columns, X_train, Y_train, preprocessor



def from_score_to_appreciation(imdb_score):
    appreciation = 'Sans avis'

    # Excellent : 6.6879 à 10 --------- 6.7 à 10
    # Bon : 6.0905 à 6.6879 ----------- 6.1 à 6.7
    # Sans avis : 5.8304 à 6.0905 ----- 5.8 à 6.1
    # Pas top : 5.4415 à 5.8304 ------- 5.4 à 5.8
    # Navet : 0 à 5.4415 -------------- 0 à 5.4

    if imdb_score >= 6.7 :
        appreciation = 'Excellent' 
    if ((imdb_score >= 6.1) and (imdb_score < 6.7)):
        appreciation = 'Bon'

    if ((imdb_score >= 5.4) and (imdb_score < 5.8)):
        appreciation = 'Pas top'
    if imdb_score < 5.4 :
        appreciation = 'Navet'

    return appreciation



def main():
    st.set_page_config(page_title="🔮 ML Movie Score",
                       page_icon="🔮",
                       layout="wide")




if __name__ == "__main__":

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["title", 
                                                    "year", 
                                                    "parentalAdvisor", 
                                                    "duree",
                                                    "resume",
                                                    "genre"])

    if "title" not in st.session_state:
        st.session_state.title = "Psycho"

    if "year" not in st.session_state:
        st.session_state.year = 1960

    if "parentalAdvisor" not in st.session_state:
        st.session_state.parentalAdvisor = "R"

    if "duree" not in st.session_state:
        st.session_state.duree = 109

    if "resume" not in st.session_state:
        st.session_state.resume = "A Phoenix secretary embezzles $40,000 from her employer's client, goes on the run, and checks into a remote motel run by a young man under the domination of his mother."

    if "genres" not in st.session_state:
        st.session_state.genres = ['Horror','Thriller']

    if "genre" not in st.session_state:
        st.session_state.genre = "Horror"

    if "score" not in st.session_state:
        st.session_state.score = 6.4

    if "themaScore" not in st.session_state:
        st.session_state.themaScore = 1.0


    with st.form(key="add form", clear_on_submit= True):

        #st.subheader(" Create your movie ")

        fcol1, fcol2 = st.columns(2)

        with fcol1: 
            st.text_input(label='🎬 Movie title :',
                          placeholder='Gone Girl, The Game, Respect... ', 
                          key='title')

            st.multiselect(label='⭐️ Genre(s) :',
                           options=['Drama', 'Comedy', 'Documentary', 'Crime', 'Romance', 'Thriller', 'Horror', 'Adventure'],
                           key='genres')

            st.text_area(label='🚢 Resume :',
                         placeholder='Describe the senario...',
                         key='resume')

        with fcol2: 
            st.number_input(label='🎟️ Release date :', 
                            min_value=1895, 
                            max_value=2022,
                            step=1, 
                            key='year')

            st.selectbox(label='⚡ TV Parental Guidelines :',
                         options=('Not Rated', 'R', 'PG', 'Approved', 'PG-13'),
                         index= 0, 
                         key='parentalAdvisor')

            st.number_input(label='🐌 Duration :',
                            min_value=1,
                            max_value=873,
                            step=1,
                            key='duree')

        # you can insert code for a list comprehension here to change the data (rwdta) 
        # values into integer / float, if required

            if st.form_submit_button("☔ Action ☔"):

                #Create Variables 
                model = load_model()
                target_name = 'imdbRating'
                dataset_columns, X_train, Y_train, preprocessor = load_data(target_name)

                
                # Input datas variables 
                data_dict = {'title': [st.session_state.title], 
                             'year': [st.session_state.year],
                             'parentalAdvisor': [st.session_state.parentalAdvisor],
                             'duree': [st.session_state.duree],
                             'resume': [st.session_state.resume],
                             'genre': [st.session_state.genres] }

                film = pd.DataFrame(data=data_dict)


                # Process 
                X_test, Y_test, my_film = test_set_from(film, target_name, dataset_columns)

                # Preprocessing 
                X_train = preprocessor.fit_transform(X_train) # Preprocessing influenceur
                X_test = preprocessor.transform(X_test) # Preprocessing copieur

                # Run Predict
                my_imdb_score = model.predict(X_test)[0]
                st.session_state.score = round(my_imdb_score,1)
                st.session_state.genre = my_film.genre[0]
                st.session_state.themaScore = my_film.themaScore[0]


    col1, col2, col3 = st.columns(3)

    st.write(
        """
        <style>
        [data-testid="stMetricDelta"] svg {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with col1:
        st.metric(label=f"{st.session_state.year}", 
                  value= f"{st.session_state.title}", 
                  delta=f"{st.session_state.duree} min", 
                  delta_color="off")

    with col2:
        scoreML = from_score_to_appreciation(st.session_state.score)
        estimation = round( st.session_state.score, 1)
        metric2 = st.metric(label="Score ML", value=f"{scoreML}", delta=f"Estimation : {estimation}/10", delta_color="off")

    with col3:
        degCos = round( np.degrees(np.arccos(st.session_state.themaScore)), 2)
        metric3 = st.metric(label="1ˢᵗ Genre", value= f"{st.session_state.genre}", delta=f"{degCos}° from resume", delta_color="off")


            

    

    st.caption('This Machine Learning script tries to guess the IMDB rating of a movie and is main genre from its resume, genres, duration, release date and parental guidelines.')
    st.caption("Projet DemoDays Jedha by [Gauthier Rammault](https://www.linkedin.com/in/gauthier-rammault/), the guy dreams to wanna be a real Data Scientist.")
