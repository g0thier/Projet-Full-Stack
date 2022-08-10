# https://spys.one/en/free-proxy-list/

import json
import logging
from operator import concat
import os
import pandas as pd
import re
import scrapy
from scrapy.crawler import CrawlerProcess

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#  Import Class Function  #
#_________________________#


class QuotesSpider(scrapy.Spider):

    # Name of your spider
    name = "iTunes"

    # Url to start your spider from 
    start_urls = ['https://spys.one/en/free-proxy-list/']

    def parse(self, response):

        quotes = response.xpath('/html/body/table[2]/tbody/tr[4]/td/table/tbody/tr')
        for quote in quotes:
            yield {
                try:
                    url = quote.xpath('td[1]/font/text()[1]/text()').get()
                    port = quote.xpath('td[1]/font/text()[2]/text()').get()
                    'proxy': url + ':' + port
                except:
                    url = ''
                    port = ''
                    'proxy': url + ':' + port
            }



#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
# Create Folder Directory #
#_________________________#


try: 
    parent_dir = os.path.abspath(os.path.split(__file__)[0])
    folder_name = "src"
    print(parent_dir)

    path = os.path.join(parent_dir, folder_name)
    os.mkdir(path)
except:
    print("File probably exists")


#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
# Create Folder Directory #
#_________________________#


# Name of the file where the results will be saved
filename = "iTunesFilmsCollection.json"

## More info on built-in settings => https://docs.scrapy.org/en/latest/topics/settings.html?highlight=settings#settings
process = CrawlerProcess(settings = {
    'USER_AGENT': 'Chrome/97.0',
    'LOG_LEVEL': logging.ERROR,
    "FEEDS": {
        path + '/' + filename : {"format": "json"},
    }
})

# Start the crawling using the spider you defined above
process.crawl(QuotesSpider)
process.start()
