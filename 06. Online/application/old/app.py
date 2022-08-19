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
from datetime import datetime

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

    return X_test, Y_test



@st.cache(persist=True, allow_output_mutation=True)
def load_model():
    model = joblib.load("src/random_forest.joblib")
    return model



@st.cache(persist=True, allow_output_mutation=True)
def load_data(target_name):
    # Import Dataset
    dataset = pd.read_csv('src/dataset.csv', delimiter=',', on_bad_lines='skip')
    dataset = dataset.drop(columns=['title', 'director', 'resume'])
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



@st.cache(persist=True, allow_output_mutation=True)
def load_film():
    film = pd.read_json('src/film.json')
    return film



#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#       create variables       #
#______________________________#

model = load_model()
target_name = 'imdbRating'
dataset_columns, X_train, Y_train, preprocessor = load_data(target_name)

my_imdb_score = 0.0 # init @ 0 

my_film = load_film()



# Process 
X_test, Y_test = test_set_from(my_film, target_name, dataset_columns)

X_train = preprocessor.fit_transform(X_train) # Preprocessing influenceur
X_test = preprocessor.transform(X_test) # Preprocessing copieur


# Run Predict
my_imdb_score = model.predict(X_test)[0]
my_imdb_score = round(my_imdb_score,1)


def main():
    st.set_page_config(page_title="🔮 IMDB Movie Score",
                       page_icon="🔮",
                       layout="wide")


if __name__ == "__main__":

    import streamlit as st

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
        # st.header()
        # st.subheader('Create your movie')
        metric1 = st.metric(label=f"{my_film['year'][0]}", 
                                    value= f"{my_film['title'][0]}", 
                                    delta=f"{round(my_film['duree'][0])} min", delta_color="off")

    with col2:
        metric2 = st.metric(label="Score IMDB", value= f"{my_imdb_score}", delta="Estimation", delta_color="off")

    with col3:
        degCos = round( np.degrees(np.arccos(my_film['themaScore'][0])), 2)
        metric3 = st.metric(label="1ˢᵗ Genre", value= f"{my_film['genre'][0]}", delta=f"{degCos}° from resume", delta_color="off")

    with st.form("my_form", clear_on_submit=False):

        fcol1, fcol2 = st.columns(2)
        
        with fcol1: 
            form_title = st.text_input('Movie title :', value=my_film['title'][0], 
                                placeholder='Gone Girl, The Game, Respect... ', key='form_title')

            form_genre = st.multiselect('Genre(s) :',
                                    ['Biography', 'Drama', 'Romance'],
                                    my_film['genre'][0], key='form_genre')

            form_resume = st.text_area('Resume to analyze :', my_film['resume'][0], key='form_resume')



        with fcol2: 
            form_year = st.number_input('Release date :', 1895, 2022, int(my_film['year'][0]), step=1, key='form_year')

            form_parentalAdvisor = st.selectbox('TV Parental Guidelines :',
                                                ('Not Rated','PG-13', 'TV-Y', 'TV-Y7', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA'),
                                                index= 0, key='form_parentalAdvisor')

            form_duree = st.number_input('Duration :', 1, 873, int(my_film['duree'][0]), step=1, key='form_duree')
                  
            submitted = st.form_submit_button("Submit")
        
        if submitted:
            export_genre = form_genre
            if type(export_genre) != list:
                export_genre = [export_genre]
                
            # Modify datas
            inputDatas = pd.DataFrame(columns=['title','year','parentalAdvisor','duree','resume','genre'],
                          data=[[form_title, form_year, form_parentalAdvisor, form_duree, form_resume, export_genre ]])

            inputDatas.to_json(r'src/film.json')

            # Format informations for predict
            X_test, Y_test = test_set_from(inputDatas, target_name, dataset_columns)
            X_test = preprocessor.transform(X_test) # Preprocessing copieur
            # Run Predict
            my_imdb_score = model.predict(X_test)[0]
            my_imdb_score = round(my_imdb_score,1)

            print(my_film)




    st.caption('This Machine Learning script tries to guess the IMDB rating of a movie and is main genre from its resume, genres, duration, release date and parental guidelines.')
    
    #st.subheader('See your result :')
    #st.dataframe(my_film)

