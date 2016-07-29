#!/usr/bin/python
# -*- coding: utf-8 -*-

###
#	University of Edinburgh, School of Informatics
#	Dissertation for the MSc Degree
#
#	BBC NEWS WEB SCRAPER
# 	Author: Paul W. Coles
#   Retrieves news stories published in a 24-hour window, stores them locally as
#   plain text by metadata category.
###

from bs4 import BeautifulSoup
import sys
import urllib2
import re
import time
import os

def get_primary_links(homepage):
    primary_links = []
    home_html = urllib2.urlopen(homepage)
    home_soup = BeautifulSoup(home_html)
    topbar = home_soup.find('div', class_ = "navigation navigation--wide")
    for a in topbar.find_all('a', href=True):
        link = a['href']
        full_link = make_long_url(link)
        if link != '/news/video_and_audio/video' and full_link not in topbar_links:
            primary_links.append(full_link)
    return primary_links


def get_secondary_links(primary_links):
    for primary_link in primary_links:
        topbar_links.append(primary_link)
        try:
            second_html = urllib2.urlopen(primary_link)
            second_soup = BeautifulSoup(second_html)
            second_bar = second_soup.find('div', class_ = "secondary-navigation secondary-navigation--wide")
            count = 0
            for a in second_bar.find_all('a', href=True):
                link = a['href']
                full_link = make_long_url(link)
                if link != '/news/video_and_audio/video' and full_link not in topbar_links:
                    topbar_links.append(full_link)
                    count += 1
                    # print "Got %i secondary links..." % count
        # Primary page may contain zero secondary links
        except:
            continue

def make_long_url(url):
    if url.startswith('http://'):
        return url
    else:
        return 'http://bbc.co.uk' + url


def make_story_soups(topbar_links):
    story_soups = {}
    stories_crawled = []
    total_homepages = len(topbar_links)
    for index, homepage in enumerate(topbar_links):
        print "\nFinding stories from homepage %i of %i: \t %s" % (index + 1, total_homepages, homepage)
        stories_found = 0
        failed_stories = 0
        out_of_window = 0
        not_a_story = 0
        homepage_html = urllib2.urlopen(homepage)
        homepage_soup = BeautifulSoup(homepage_html)
        homepage_links = homepage_soup.find_all('a', href=True)
        links_count = len(homepage_links)
        print "%i links to check..." % links_count
        for index, a in enumerate(homepage_links):
            # print "Homepage: %s \t Checking link %i of %i..." % (homepage, index + 1, links_count)
            long_url = make_long_url(a['href'])
            # Of all links on page, make soup only for the story pages
            if "/news/" in long_url and re.search(r"\d{8}", long_url) is not None \
            and "/live/" not in long_url and long_url not in stories_crawled:
                try:
                    story_html = urllib2.urlopen(long_url)
                    story_soup = BeautifulSoup(story_html)
                    try:
                        v1 = story_soup.find(['p','div'], class_ = 'date date--v1')
                        v2 = story_soup.find('div', class_ = 'date date--v2')
                        if v1 is not None:
                            date_v1 = v1['data-seconds']
                            published = int(date_v1)
                        elif v2 is not None:
                            date_v2 = v2['data-seconds']
                            published = int(date_v2)
                    except:
                        print "No date found"
                        continue
                except:
                    print "Can't make soup."
                    continue

                try:
                    if published >= window_start and published <= window_end:
                        story_soups[long_url] = story_soup
                        stories_crawled.append(long_url)
                        stories_found += 1
                        # print "Story success: \t" + long_url
                    else:
                        # print "Story out of window: \t" + long_url
                        out_of_window += 1
                except:
                    # print "Failed story: \t" + long_url
                    failed_stories += 1
                    continue
            else:
                not_a_story += 1
        print "\nSummary for homepage:\t %s" % homepage
        print "%i stories found\n%i stories out of window\n%i links are not stories\
        \n%i stories failed\n\n" % (stories_found, out_of_window, not_a_story, failed_stories)


        # Avoid scraping BBC server too frequently
        time.sleep(0.1)
    return story_soups


def write_story(url, story_soup, out_directory):
    # Get metadata
    category = str(story_soup.find('meta', attrs = {'property':'og:article:section'})['content'])
    category = re.sub(' ', '_', category)           # Use published metadata for category
    story_id = (url.split('/'))[-1].split('-')[-1]  # Integer ID from the URL
    # Get body text
    try:
        try:
            tag = story_soup.find("div", {"class":"story-body__inner"})
            p_tags = tag.find_all('p')
            print "Archiving text page: \t" + url
        except:
            tag = story_soup.find("div", {"class":"text-wrapper"})
            p_tags = tag.find_all('p')
            print "Archiving video page: \t" + url
        body = ''
        for i in p_tags:
            if i.string:
                i = re.sub(r'<.+>',' ', i.string)
                body += ' ' + i
        body = body.encode(encoding='UTF-8',errors='replace')
    # Make category directory
        cat_directory = out_directory + '/' + category + '/'
        if not os.path.exists(cat_directory):
            os.makedirs(cat_directory)
        with open(cat_directory + story_id + '.txt', 'w') as f:
            f.write(body)
    except:
        print "No text found for: \t" + url
        pass

if __name__ == "__main__":
    # Set story time window for this run
    if len(sys.argv) == 2:
        time_mode = sys.argv[1]
    else:
        print "Usage: bbc_scrape.py [early, late, rel]"
        sys.exit(1)

    localtime = time.localtime(time.time())
    # Absolute window, early
    if time_mode == 'early':
        t_early = localtime[0:3] + (5, 59, 59) + localtime[6:]
        window_end = int(time.mktime(t_early))
        window_start = window_end - 86399
    # Absolute window, late
    elif time_mode == 'late':
        t_late = localtime[0:3] + (17, 59, 59) + localtime[6:]
        window_end = int(time.mktime(t_late))
        window_start = window_end - 86399
    # Relative window
    elif time_mode == 'rel':
        window_end = int(time.mktime(localtime))
        window_start = window_end - 86399
    else:
        print 'Invalid time mode.'
        sys.exit(1)

    # Get today's navigation bar links
    topbar_links = []
    print "Getting primary links... "
    primary_links = get_primary_links("http://www.bbc.co.uk/news")
    print "Got %i primary links." % len(primary_links)
    print "\nGetting secondary links... "
    get_secondary_links(primary_links)

    # Test
    # topbar_links = ['http://www.bbc.co.uk/news/world/']

    # Store BeautifulSoup instances for each story page
    story_soups = make_story_soups(topbar_links)
    print "\n"

    # Make a directory for today's output
    year, month, day = time.strftime("%Y"), time.strftime("%m"), time.strftime("%d")
    out_directory = '../data/' + year + '/' + month + '/' + day + '/' + time_mode + '/'
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # Use locally-stored instances to find article text and write to file
    for url, story_soup in story_soups.iteritems():
        write_story(url, story_soup, out_directory)
