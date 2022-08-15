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

    # Convert string Genre to list
    my_film["genres"] = [ str(x).replace(' ', '') for x in my_film["genres"][:] ] 
    my_film["genres"] = [ x.split(",") for x in my_film["genres"][:] ]

    model = SentenceTransformer('all-MiniLM-L6-v2')

    my_film['genre'] = ''
    my_film['themaScore'] = 0.0

    for i in range( len(my_film) ):
        resume = my_film.iloc[i]['resume']
        genres = my_film.iloc[i]['genres']
        bestScore = 0.0
        bestGenre = ''
        for genre in genres:
            emb1 = model.encode(str(genre))
            emb2 = model.encode(str(resume))
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



@st.cache(persist=True)
def load_model():
    model = joblib.load("src/random_forest.joblib")
    return model



@st.cache(persist=True)
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



#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#       create variables       #
#______________________________#

model = load_model()

target_name = 'imdbRating'

dataset_columns, X_train, Y_train, preprocessor = load_data(target_name)

my_imdb_score = 0.0 # init @ 0 


title = 'Radioactive'
year = 2019
parentalAdvisor = 'PG-13'
duree = 109.0
resume = 'The incredible true story of Marie Sklodowska-Curie and her Nobel Prize-winning work that changed the world.'
genre = ['Biography', 'Drama', 'Romance']

my_film = pd.DataFrame(columns=['title','year','parentalAdvisor','duree','resume','genre'],
                          data=[[ title , year , parentalAdvisor , duree , resume , genre ]])


X_test, Y_test = test_set_from(my_film, target_name, dataset_columns)

X_train = preprocessor.fit_transform(X_train) # Preprocessing influenceur
X_test = preprocessor.transform(X_test) # Preprocessing copieur


# Run Predict
my_imdb_score = model.predict(X_test)[0]
my_imdb_score = round(my_imdb_score,1)


st.set_page_config(layout="wide")

st.header('🔮 IMDB Movie Score')
st.subheader('Create your movie')
st.caption('This Machine Learning script guesses the IMDB rating of a movie from its resume, genre, duration, release date and parental guidelines')


with st.form("my_form", clear_on_submit=False):
    st.write("Inside the form")
    slider_val = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox")
    form_title = st.text_input('Movie title :', value=my_film['title'][0], 
                               placeholder='Gone Girl, The Game, Respect... ')

    form_year = st.slider('Release date :', 1895, 2022, my_film['year'][0])
    form_parentalAdvisor = st.selectbox('TV Parental Guidelines',
                                        ('PG-13', 'TV-Y', 'TV-Y7', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA'),
                                        index= 0 )
    
    form_duree = st.slider('Duration :', 1, 873, my_film['duree'][0])

    form_resume = st.text_area('Resume to analyze :', my_film['resume'][0])

    form_genre = st.multiselect('Genre(s) :',
                                ['Biography', 'Drama', 'Romance'],
                                my_film['genre'][0])

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        my_film['title'][0] = form_title
        my_film['year'][0] = form_year
        my_film['parentalAdvisor'][0] = form_parentalAdvisor
        my_film['duree'][0] = form_duree
        my_film['resume'][0] = form_resume
        my_film['genre'][0] = form_genre
        # Format informations for predict
        X_test, Y_test = test_set_from(my_film, target_name, dataset_columns)
        X_test = preprocessor.transform(X_test) # Preprocessing copieur
        # Run Predict
        my_imdb_score = model.predict(X_test)[0]
        my_imdb_score = round(my_imdb_score,1)
        st.write("slider", slider_val, "checkbox", checkbox_val)


st.subheader('See your result :')
st.dataframe(my_film)

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Score IMDB", value= f"{my_imdb_score}", delta="Estimation", delta_color="off")

with col2:
    st.metric(label="1ᵉʳ Genre", value= f"{my_film['genre'][0]}", delta="Estimation", delta_color="off")

