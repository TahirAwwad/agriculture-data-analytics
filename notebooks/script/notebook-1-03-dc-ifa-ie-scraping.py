#!/usr/bin/env python
# coding: utf-8

# ## Parse Irish Farmers Association articles

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("tahirawwad", "agriculture-data-analytics", "notebooks/notebook-1-03-dc-ifa-ie-scraping.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/tahirawwad/agriculture-data-analytics/master?filepath=notebooks/notebook-1-03-dc-ifa-ie-scraping.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/tahirawwad/agriculture-data-analytics/blob/master/notebooks/notebook-1-03-dc-ifa-ie-scraping.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective

# The objective is to download the contents of the articles on the [Irish Farmers Association](https://www.ifa.ie/) and create a dataset.  

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source file paths.

# Local
#!pip install -r script/requirements.txt
# Remote option
#!pip install -r https://raw.githubusercontent.com/tahirawwad/agriculture-data-analytics/requirements.txt
#Options: --quiet --user


from bs4 import BeautifulSoup
from pandas import DataFrame
from requests.sessions import Session
import requests


# ### Functions

session: Session = requests.Session()

session.headers = {
    "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
    "Accept-Encoding": "gzip, deflate",
    "Accept":
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/png,*/*;q=0.8",
    "Accept-Language": "en"
}


def get_articles_html(term: str = "cattle", pages=1000):
    query_post_data: dict = {
        'query':
        f'{{"sector":"{term}","error":"","m":"","p":0,"post_parent":"","subpost":"","subpost_id":"","attachment":"","attachment_id":0,"name":"","pagename":"","page_id":0,"second":"","minute":"","hour":"","day":0,"monthnum":0,"year":0,"w":0,"category_name":"","tag":"","cat":"","tag_id":"","author":"","author_name":"","feed":"","tb":"","paged":0,"meta_key":"","meta_value":"","preview":"","s":"","sentence":"","title":"","fields":"","menu_order":"","embed":"","category__in":[],"category__not_in":[],"category__and":[],"post__in":[],"post__not_in":[],"post_name__in":[],"tag__in":[],"tag__not_in":[],"tag__and":[],"tag_slug__in":[],"tag_slug__and":[],"post_parent__in":[],"post_parent__not_in":[],"author__in":[],"author__not_in":[],"ignore_sticky_posts":false,"suppress_filters":false,"cache_results":true,"update_post_term_cache":true,"lazy_load_term_meta":true,"update_post_meta_cache":true,"post_type":"","posts_per_page":130,"nopaging":false,"comments_per_page":"50","no_found_rows":false,"taxonomy":"sector","term":"{term}","order":"DESC"}}',
        'action': 'loadmore',
        'page': '',
    }
    url: str = "https://www.ifa.ie/wp-admin/admin-ajax.php/"

    page_number: int = 0
    html: str = ""
    receiving_data: bool = True
    while receiving_data and page_number < pages:
        query_post_data['page'] = page_number
        text: str = session.post(url, query_post_data).text
        if not text: receiving_data = False
        else: html += text
        page_number += 1

    print(f"Downloaded {page_number} pages.")
    return html


def get_articles_links(html: str):
    beautiful_soup = BeautifulSoup(html, 'html.parser')
    links = beautiful_soup.find_all("a", {"class": ""}, href=True)
    print("Article Links found:", len(links))
    return links


def download_articles_from_links(links, term: str):
    page_list: list = []

    for link in links:
        url: str = link['href']
        response: str = session.get(url)
        beautiful_soup = BeautifulSoup(response.content, 'html.parser')
        heading = beautiful_soup.find('h1').text
        date = beautiful_soup.find('time').text
        html_content = beautiful_soup.find("div", {"class": "single-content"})
        page_list.append(
            [url, heading, date, term, html_content.text, html_content])
    return page_list


def append_articles_to_csv(page_list, filename: str):
    APPEND = 'a'
    dataframe_columns = [
        "URL", "Heading", "Date", "HTML Content", "Text", "Trend"
    ]
    dataframe = DataFrame(page_list, columns=dataframe_columns)
    dataframe.to_csv(f'./../assets/{filename}',
                     index=False,
                     header=False,
                     mode=APPEND)
    print(f"Dataframe saved to assets/{filename}")


def create_articles_csv_file(filename: str):
    dataframe_columns = [
        "URL", "Heading", "Date", "Trend", "Text", "HTML Content"
    ]
    dataframe = DataFrame(columns=dataframe_columns)
    dataframe.to_csv(f'./../assets/{filename}', index=False)
    print(f"Created assets/{filename}")


def download_articles(term: str, filename: str) -> None:
    html = get_articles_html(term)
    links = get_articles_links(html)
    page_list = download_articles_from_links(links, term)
    append_articles_to_csv(page_list, filename)


# ### Save Artifacts

# Saving the output of the notebook.

filename = f"ifa-ie-articles.csv"
create_articles_csv_file(filename)
terms = ["cattle", "dairy"]
for term in terms:
    download_articles(term, filename)


# Author &copy; 2021 <a href="https://github.com/markcrowe-com" target="_parent">Mark Crowe</a>. All rights reserved.
