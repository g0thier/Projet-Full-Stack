import pandas as pd
import numpy as np
import os

# reconstitution du chemin 
parent_dir = os.path.abspath(os.path.split(__file__)[0])
folder_name = "src"
file_name = "lieux-de-tournage-a-paris.csv"

path = os.path.join(parent_dir, folder_name, file_name)


df_tournage = pd.read_csv(path, delimiter=';', on_bad_lines='skip')

# Drop unused :
df_tournage = df_tournage.drop(columns=["Identifiant du lieu","Localisation de la scène","Année du tournage","Coordonnée en X", "Coordonnée en Y","geo_shape"])

# Corrections Type :
df_tournage["Code postal"] = pd.to_numeric(df_tournage["Code postal"])
df_tournage = df_tournage.dropna(subset = ['Code postal'])
df_tournage["Code postal"] = df_tournage["Code postal"].astype(int)
df_tournage["Date de début"] = pd.to_datetime(df_tournage["Date de début"])
df_tournage["Date de fin"] = pd.to_datetime(df_tournage["Date de fin"])
df_tournage["latitude"] = pd.to_numeric( [ x.split(",")[0] for x in df_tournage["geo_point_2d"][:][:] ] )
df_tournage["longitude"] = pd.to_numeric( [ x.split(",")[1] for x in df_tournage["geo_point_2d"][:][:] ] )
df_tournage = df_tournage.drop(columns="geo_point_2d")

# Clear Name Columns :
df_tournage = df_tournage.rename(columns = {"Type de tournage":"type", "Code postal":"postal", "Date de début":"debut", "Date de fin":"fin"})
df_tournage.columns = [ x.lower() for x in df_tournage.columns.tolist() ] # ABC -> abc

# keep only films :
df_tournage = df_tournage[df_tournage["type"] == 'Long métrage']
df_tournage = df_tournage.drop(columns= ['type'])

df_tournage = df_tournage.reset_index().drop(columns=['index'])


#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#      Create Links       #
#_________________________#


import urllib.parse

url_base = 'https://www.imdb.com/search/title/?title='
param2 = '&release_date=2015-01-01,2022-12-31'

df_tournage['imdb_search'] = ''
# https://www.imdb.com/search/title/?title=TOUT+S%27EST+BIEN+PASSE&release_date=2015-01-01,2022-12-31

for i in range( len(df_tournage) ):

    title = urllib.parse.quote(df_tournage.iloc[i]['titre']).replace('%20', '+')

    df_tournage.imdb_search[i] = url_base + title + param2


#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#     Init Scraping       #
#_________________________#

import json
import logging
import os
from turtle import title
import pandas as pd
import re
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy_splash import SplashRequest

# reconstitution du chemin 
parent_dir = os.path.abspath(os.path.split(__file__)[0])
folder_name = "export"

path = os.path.join(parent_dir, folder_name)


#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#     Class Scraping      #
#_________________________#

class QuotesSpider(scrapy.Spider):

    # Name of your spider
    name = "imdb"

    # Url to start your spider from 
    start_urls = df_tournage['imdb_search'].tolist()[:]

    # Rotation de proxy 
    ROTATING_PROXY_LIST_PATH = '/my/path/proxies.txt' # Path that this library uses to store list of proxies
    NUMBER_OF_PROXIES_TO_FETCH = 5 # Controls how many proxies to use

    # Recherche Requête 
    def parse(self, response):

        print('HTTP status normal 200 :')
        print(response.status)

        quotes = response.xpath('//*[@id="main"]/div/div[3]/div/div[1]/div[3]')

        for quote in quotes:
            
            # Requete
            try:
                imdb_search = response.url
            except:
                imdb_search = ''
            # title
            try:
                title = quote.xpath('h3/a/text()').get()
            except:
                title = ''
            # date
            try:
                date = quote.xpath('h3/span[2]/text()').get()
            except:
                date = ''
            # categorie Age
            try:
                parental_advisor = quote.xpath('p[1]/span[1]/text()').get()
            except:
                parental_advisor = ''
            # duree
            try:
                duree = quote.xpath('p[1]/span[3]/text()').get()
            except:
                duree = ''
            # Genre
            try:
                genre = quote.xpath('p[1]/span[5]/text()').get()
            except:
                genre = ''
            # Note IMDB 
            try: 
                imdb_note = quote.xpath('div/div[1]/strong/text()').get()
            except: 
                imdb_note = ''
            # Metascore 
            try:
                metascore = quote.xpath('div/div[3]/span/text()').get()
            except:
                metascore = ''
            # Votes
            try:
                vote = quote.xpath('p[4]/span[2]/text()').get()
            except:
                vote = ''
            # resume 
            try:
                resume = quote.xpath('p[2]/text()').get()
            except:
                resume = ''

            yield {
                'imdb_search' : imdb_search,
                'title' : title,
                'date' : date,
                'parental_advisor' : parental_advisor,
                'duree' : duree,
                'genre' : genre,
                'imdb_note' : imdb_note,
                'metascore' : metascore,
                'vote' : vote,
                'resume' : resume,

            }


#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#     Start Scraping      #
#_________________________#

# Name of the file where the results will be saved
filename = "imdb_films_infos.json"

# Settings Crawler 
process = CrawlerProcess(settings = {
    'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246',
    'LOG_LEVEL': logging.ERROR,
    "FEEDS": {
        path + '/' + filename : {"format": "json"},
    }
})

# Start the crawling using the spider you defined above
process.crawl(QuotesSpider)
process.start()