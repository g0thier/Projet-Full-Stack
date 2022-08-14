import pandas as pd
import numpy as np
import os
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
file_name = "dataset_without_resume.csv"
path = os.path.join(parent_dir, folder_name, file_name)
dataset = pd.read_csv(path, delimiter=',', on_bad_lines='skip')


#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#     Class Scraping      #
#_________________________#

class QuotesSpider(scrapy.Spider):

    # Name of your spider
    name = "imdb"

    # Url to start your spider from 
    start_urls = dataset['imdb_search'].tolist()[:]

    # Delay for don't be desallow 
    custom_settings = {
        'AUTOTHROTTLE_ENABLED': True, # limite pour ménager les serveur 
        'AUTOTHROTTLE_DEBUG': True, # affiche le debug
        'DOWNLOAD_DELAY': 0.1, # temps entre les requetes 
        'DEPTH_LIMIT': 1, # profondeur de recherche 
    }

    # Rotation de proxy 
    ROTATING_PROXY_LIST_PATH = path + '/proxies.txt' # Path that this library uses to store list of proxies
    NUMBER_OF_PROXIES_TO_FETCH = 25 # Controls how many proxies to use

    # paralelle
    DOWNLOADER_MIDDLEWARES = {
        'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
        'scrapy_user_agents.middlewares.RandomUserAgentMiddleware': 400,
        #'rotating_proxies.middlewares.RotatingProxyMiddleware': 610,
        #'rotating_proxies.middlewares.BanDetectionMiddleware': 620,
        'rotating_free_proxies.middlewares.RotatingProxyMiddleware': 610,
        'rotating_free_proxies.middlewares.BanDetectionMiddleware': 620,
    }

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
            # resume 
            try:
                resume = quote.xpath('p[2]/text()').get()
            except:
                resume = ''

            yield {
                'imdb_search' : imdb_search,
                'parental_advisor' : parental_advisor,
                'duree' : duree,
                'genre' : genre,
                'resume' : resume,

            }


#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#     Start Scraping      #
#_________________________#

path = os.path.join(parent_dir, folder_name)
filename = "imdb_resumes.json"

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