#!/usr/bin/env python
# coding: utf-8

# ## Download Twitter Data

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("tahirawwad", "agriculture-data-analytics", "notebooks/notebook-1-04-dc-twitter-api.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/tahirawwad/agriculture-data-analytics/master?filepath=notebooks/notebook-1-04-dc-twitter-api.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/tahirawwad/agriculture-data-analytics/blob/master/notebooks/notebook-1-04-dc-twitter-api.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Setup
# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt
# Remote option
#!pip install -r https://raw.githubusercontent.com/tahirawwad/agriculture-data-analytics/requirements.txt
#Options: --quiet --user


from configparser import ConfigParser
from pandas import DataFrame
import csv
import pandas
import tweepy


# ### Twitter
# #### Irish Farming Accounts
# 
# Dept of Agriculture, Food and the Marine:  <a href="https://twitter.com/agriculture_ie" target="_new">@agriculture_ie</a>  
# Leading the sustainable development of Ireland’s agri-food, forestry and marine sectors.  
# 
# Irish Farmers' Association: <a href="https://twitter.com/ifamedia" target="_new">@ifamedia</a>  
# IFA has protected and defended the interests of Irish farmers and their families at home and in Europe for more than 65 years.  
# 
# Irish Farmers Journal : <a href="https://twitter.com/farmersjournal" target="_new">@farmersjournal</a>  
# Ireland's largest selling farming & rural living publication | Readership: 321,400 weekly print readers on the Island of Ireland. Source: Kantar 2021 TGI Survey
# 
# <a href="https://twitter.com/teagasc" target="_new">@Teagasc</a>  
# Teagasc - Providing research, advisory & training services to the agriculture and food industry & rural communities.
# 
# <a href="https://twitter.com/thatsfarming" target="_new">@thatsfarming.com</a>  
# Latest news on all things Farming. If you send us a tweet, you consent to letting http://Thatsfarming.com use & showcase it in any media.
# 
# <a href="https://twitter.com/agrilandireland" target="_new">@AgrilandIreland</a>  
# Ireland’s largest farming news portal. Daily updates covering all agricultural sectors.
# 
# #### US and World Accounts
# 
# <a href="https://twitter.com/USDA" target="_new">@USDA</a> (131,000+ followers). Stay up-to-date with the latest news and information from the USDA, including links to the USDA Blog, recent videos and commentary from Secretary of Agriculture Tom Vilsack. @USDA also live tweets from industry events.
# 
# <a href="https://twitter.com/MonsantoCo" target="_new">@MonsantoCo</a> (24,000+). Monsanto provides regular updates on the company’s latest innovation and news, as well as access to the company’s blog, Beyond the Rows. The blog is updated by Monsanto employees who write about the company’s business, the ag industry, the farmer and more.
# 
# <a href="https://twitter.com/nationalffa" target="_new">@nationalffa</a> (24,000+). Formerly the Future Farmers of America, the National FFA Organization uses Twitter as a great way to keep its followers (and members) current on activities such as awards programs, outreach efforts and career opportunities.
# 
# <a href="https://twitter.com/farmbureau" target="_new">@farmbureau</a> (18,000+). American Farm Bureau is the nation’s largest farm organization, comprised of and directed by farm and ranch families who engage in all types of food and fiber production. @farmbureau gives followers access to its newsletter (Foodie News) and podcast (FoodieCast), which offer the latest in food trends.
# 
# <a href="https://twitter.com/farmingfirst" target="_new">@farmingfirst</a> (16,500+). Farming First is a global coalition that calls on world leaders to increase agricultural output in a sustainable and socially responsible manner. Many tweets include links to content found on FarmingFirst.org, which features blogs, case studies, video and more.
# 
# <a href="https://twitter.com/rodaleinstitute" target="_new">@rodaleinstitute</a> (15,500+). A leader in the organics industry since 1947, Rodale Institute has a strong following on Twitter. The institute shares its latest research on the best practices of organic agriculture with farmers and scientists throughout the world.

config_filepath = "config.ini"
config_parser = ConfigParser()
config_parser.read(config_filepath)

access_token = config_parser["Twitter"]["AccessToken"]
access_token_secret = config_parser["Twitter"]["AccessTokenSecret"]
consumer_key = config_parser["Twitter"]["ApiKey"]
consumer_secret = config_parser["Twitter"]["ApiKeySecret"]


# <https://python-twitter.readthedocs.io/en/latest/getting_started.html>  
# 
# 
# You need to have a developer account: <https://developer.twitter.com/en/portal/petition/essential/basic-info>
# 
# And apply for elevated access.
# <https://developer.twitter.com/en/portal/products/elevated>

o_auth_handler = tweepy.OAuthHandler(consumer_key, consumer_secret)
o_auth_handler.set_access_token(access_token, access_token_secret)
tweepy_api = tweepy.API(o_auth_handler, wait_on_rate_limit=True)


#Dept of Agriculture, Food and the Marine
screen_name = "agriculture_ie"


tweets = tweepy_api.user_timeline(
    screen_name=screen_name,
    count=200,  # 200 is the maximum allowed count
    include_rts=False,
    tweet_mode="extended"
)  # Necessary to keep full_text otherwise only the first 140 words are extracted


for info in tweets[:3]:
    print("ID: {}".format(info.id))
    print(info.created_at)
    print(info.full_text)
    print("\n")


#extract additional tweets

all_tweets = []
all_tweets.extend(tweets)
oldest_id = tweets[-1].id
while True:
    tweets = tweepy_api.user_timeline(
        screen_name=screen_name,
        count=200,# 200 is the maximum allowed count
        include_rts=False,
        max_id=oldest_id - 1,
        # Necessary to keep full_text
        # otherwise only the first 140 words are extracted
        tweet_mode='extended')
    if len(tweets) == 0:
        break
    oldest_id = tweets[-1].id
    all_tweets.extend(tweets)
    print('N of tweets downloaded till now {}'.format(len(all_tweets)))


tweets_list: list = [[
    tweet.id_str, tweet.user.screen_name, tweet.created_at,
    tweet.favorite_count, tweet.retweet_count,
    tweet.full_text.encode("utf-8").decode("utf-8")
] for idx, tweet in enumerate(all_tweets)]


tweet_columns = [
    "id", "screen_name", "created_at", "favorite_count", "retweet_count",
    "text"
]
dataframe = DataFrame(tweets_list, columns=tweet_columns)
dataframe.to_csv('./../assets/twitter-agriculture-ie.csv', index=False)
dataframe.head(3)

