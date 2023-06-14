
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import csv
import time
import json
from bs4.element import Tag
import numpy as np
import cloudscraper

# initialize cloudscraper, which fill help to bypass Cloudflare's anti-bot page
scraper = cloudscraper.create_scraper()

# create csv file to store news
file_to_store_news = 'news_database.csv'
with open(file_to_store_news, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['category', 'title', 'date', 'n_views', 'n_shares', 'summary', 'content', 'tags'])

# recognize paragraphs and remove "related" part of the content
def get_nice_text(soup): 
    txt = ''
    for par in soup.find_all(lambda tag:tag.name=="p" and not "Related:" in tag.text):
        txt += ' ' + re.sub(" +|\n|\r|\t|\0|\x0b|\xa0",' ',par.get_text())
    return txt.strip()

# create a DataFrame for the news
def prepare_pandas(df):
    df.index = df.date
    df.drop(columns = 'date', inplace = True)
    df.index = pd.to_datetime(df.index, utc = True)
    df.sort_index(inplace = True)
    return df

# configure headers
headers = {'Cookie':'_gcar_id=0696b46733edeac962b24561ce67970199ee8668', 
           'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

# set starting scraping statistics values for the loop
url_base = "https://cointelegraph.com/post-sitemap-"
total_posts = 0

bad_response = []
bad_response_count = 0

unparsable_webpage = []

# the code requires number of content-aggregating pages from https://cointelegraph.com/sitemap.xml
def get_n_agg_pages(headers):

    """Get number of content-aggregating pages from https://cointelegraph.com/sitemap.xml."""

    sitemap_url = 'https://cointelegraph.com/sitemap.xml'
    sitemap_webpage = scraper.get(sitemap_url, headers=headers)
    print(sitemap_webpage.status_code)
    
    sitemap_soup = BeautifulSoup(sitemap_webpage.text, features = 'xml') # parse the sitemap
    sitemap_all_links = sitemap_soup.find_all('loc') # get all links from the sitemap
    sitemap_last_part= [link.getText().split('/')[-1] for link in sitemap_all_links] # get the last part of the link
    sitemap_pages = [a for a in sitemap_last_part if a.startswith('post-sitemap-')] # get only links to content-aggregating pages
    n_agg_pages = max([int(a.split('-')[-1]) for a in sitemap_pages]) # get the number of content-aggregating pages
    return n_agg_pages


n_agg_pages = get_n_agg_pages(headers)
print(n_agg_pages)

# loop over all content-aggregating pages
for i in range(1,n_agg_pages+1): 
    url = url_base+str(i)
    print('scrapping ', url)
    web_map = scraper.get(url, headers = headers) # get the page
    # scrape all links from the page
    print("Scraping page", i, "of", n_agg_pages, "pages")   
    soup = BeautifulSoup(web_map.text, features = 'lxml') # parse the page
    all_links = soup.find_all('loc') # get all links from the page

    posts_downloaded = 0

    # loop over all links on the page
    for j, item in enumerate(all_links):
        print("Scraping post", j, "of", i, "-th page posts")
        url_post = item.getText() # get the link
        is_news = url_post.split('/')[3] # check if the link is to a news item
        
        if is_news != "news": # if the link is not to a news item, skip it
            print('\n')
            print(is_news, 'not a news item \n')
            continue

        # scrape the post
        page = scraper.get(url_post, headers = headers) # get the page
        page.encoding = 'utf-8' # set encoding
        sauce = BeautifulSoup(page.text,"lxml") # parse the page
        
        try:
            # get the data from the post
            data = json.loads(sauce.find('script', type='application/ld+json').string)
        except:
            # if the post is not available, try again
            print('Something is wrong: status', page.status_code, 'will sleep and retry')
            time.sleep(4)
            try: 
                # get the data from the post
                data = json.loads(sauce.find('script', type='application/ld+json').string)
            except:
                # skip the post if it is not available
                print('Sleeping didnt solve the problem, going to the next post')
                bad_response.append(url_post)
                bad_response_count +=1
                continue
                
        # get the article section and publication date
        try:
            art_tag = data['articleSection']    
        except: 
            art_tag = None
        try:
            date = data['datePublished']
        except:
            date = None

        # some articles have tags which could help with classification
        titleTag = sauce.find("h1",{"class":"post__title"})
        summaryTag = sauce.find("p", {"class":"post__lead"})
        contentTag = sauce.find("div",{"class":"post-content"})
        tagsTag = sauce.find('ul', {"class":"tags-list__list"}) 
        
        title = None
        content = None
        summary = None
        tags_list = None
        
        # get the data from the tags if they are present in the post
        if isinstance(titleTag,Tag):
            title = titleTag.get_text().strip()
            
        if isinstance(contentTag,Tag):
            content = get_nice_text(contentTag)
    
        if isinstance(summaryTag, Tag):
            summary = summaryTag.get_text().strip() 
            
        if isinstance(tagsTag, Tag):
            tags_str = tagsTag.get_text().strip()
            tags_list_prep = tags_str.split('#')
            tags_list = [i.strip() for i in tags_list_prep if len(i)>0]
        
        # get the number of views and shares
        stats = sauce.find_all('div', {"class" : "post-actions__item post-actions__item_stat"}) 
        
        if len(stats)>0: # if the post has views and shares, get the data
            views = stats[0]    
            views_list = views.get_text().strip().split(" ")
            count_views = int(views_list[0])
            
            if len(stats)>1:  
                shares = stats[1]
                shares_list = shares.get_text().strip().split(" ")
                count_shares = int(shares_list[0])
            else: 
                count_shares = None
        else: 
            count_views = None
            count_shares = None
                 
        # save the data to the .csv file
        with open(file_to_store_news, 'a', encoding = 'utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([art_tag, title, date, count_views, count_shares, summary, content, tags_list])
            
        posts_downloaded +=1
        
    total_posts += posts_downloaded
    print('loaded ', total_posts, 'posts')
    
    # sleep for a random time to avoid being blocked
    to_sleep = abs(np.random.normal(2, 3))
    time.sleep(to_sleep)
    
news_map = pd.read_csv(file_to_store_news)
news_map = prepare_pandas(news_map)

