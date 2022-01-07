#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import tweepy
import csv #Import csv
from pandas import DataFrame


# Irish Accounts
# 
# Dept of Agriculture, Food and the Marine
# @agriculture_ie
# Leading the sustainable development of Ireland’s agri-food, forestry and marine sectors.
# 
# Irish Farmers' Association
# @IFAmedia
# IFA has protected and defended the interests of Irish farmers and their families at home and in Europe for more than 65 years. 
# 
# Irish Farmers Journal
# @farmersjournal
# Ireland's largest selling farming & rural living publication | Readership: 321,400 weekly print readers on the Island of Ireland. Source: Kantar 2021 TGI Survey
# 
# Teagasc
# @teagasc
# Teagasc – Providing research, advisory & training services to the agriculture and food industry & rural communities.
# 
# thatsfarming.com
# @thatsfarming
# Latest news on all things Farming. If you send us a tweet, you consent to letting http://Thatsfarming.com use & showcase it in any media.
# 
# Agriland
# @AgrilandIreland
# Ireland’s largest farming news portal. Daily updates covering all agricultural sectors.
# 
# 
# 
# 
# US and World
# 
# 
# @USDA (131,000+ followers). Stay up-to-date with the latest news and information from the USDA, including links to the USDA Blog, recent videos and commentary from Secretary of Agriculture Tom Vilsack. @USDA also live tweets from industry events.
# 
# @MonsantoCo (24,000+). Monsanto provides regular updates on the company’s latest innovation and news, as well as access to the company’s blog, Beyond the Rows. The blog is updated by Monsanto employees who write about the company’s business, the ag industry, the farmer and more.
# 
# @nationalffa (24,000+). Formerly the Future Farmers of America, the National FFA Organization uses Twitter as a great way to keep its followers (and members) current on activities such as awards programs, outreach efforts and career opportunities.
# 
# @farmbureau (18,000+). American Farm Bureau is the nation’s largest farm organization, comprised of and directed by farm and ranch families who engage in all types of food and fiber production. @farmbureau gives followers access to its newsletter (Foodie News) and podcast (FoodieCast), which offer the latest in food trends.
# 
# @farmingfirst (16,500+). Farming First is a global coalition that calls on world leaders to increase agricultural output in a sustainable and socially responsible manner. Many tweets include links to content found on FarmingFirst.org, which features blogs, case studies, video and more.
# 
# @rodaleinstitute (15,500+). A leader in the organics industry since 1947, Rodale Institute has a strong following on Twitter. The institute shares its latest research on the best practices of organic agriculture with farmers and scientists throughout the world.
# 
# 
# 

consumer_key = input('enter your token')
consumer_secret = input('enter your token')
access_token = input('enter your token')
access_token_secret = input('enter your token')
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)


#Dept of Agriculture, Food and the Marine
#Leading the sustainable development of Ireland’s agri-food, forestry and marine sectors.
userID = "agriculture_ie"


tweets = api.user_timeline(screen_name=userID, 
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended'
                           )


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
    tweets = api.user_timeline(screen_name=userID, 
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                           max_id = oldest_id - 1,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended'
                           )
    if len(tweets) == 0:
        break
    oldest_id = tweets[-1].id
    all_tweets.extend(tweets)
    print('N of tweets downloaded till now {}'.format(len(all_tweets)))


# outtweets = [[tweet.id_str, 
#               tweet.created_at, 
#               tweet.favorite_count, 
#               tweet.retweet_count, 
#               tweet.full_text.encode("utf-8").decode("utf-8")] 
#              for idx,tweet in enumerate(all_tweets)]
# df = DataFrame(outtweets,columns=["id","created_at","favorite_count","retweet_count", "text"])
# df.to_csv('%s

outtweets = [[tweet.id_str, 
              tweet.created_at, 
              tweet.favorite_count, 
              tweet.retweet_count, 
              tweet.full_text.encode("utf-8").decode("utf-8")] 
             for idx,tweet in enumerate(all_tweets)]
df = DataFrame(outtweets,columns=["id","created_at","favorite_count","retweet_count", "text"])
df.to_csv('./../artifacts/agriculture_ie.csv',index=False)
df.head(3)




