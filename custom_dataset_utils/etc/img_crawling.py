from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import requests as req
import time
from selenium.webdriver.common.keys import Keys
from urllib.request import urlopen
import urllib
from bs4 import BeautifulSoup
import random
import os
import sys


browser = webdriver.Chrome(ChromeDriverManager().install())
browser.implicitly_wait(3)

count = 0
size = 30
search_word = "군복"

photo_list = []
target_dir = "/media/daton/D6A88B27A88B0569/dataset/military/images"

params = {
    "q": search_word,
    "tbm": "isch",
    "sa": "1",
    "source": "lnms&tbm=isch"
}
url = "https://www.google.com/search"
url = url + "?" + urllib.parse.urlencode(params)
time.sleep(0.5)

browser.get(url)
html = browser.page_source
time.sleep(0.5)

soup_temp = BeautifulSoup(html, "html.parser")
img4page = len(soup_temp.findAll("img"))

elem = browser.find_element_by_tag_name("body")
while count < size * 10:
    elem.send_keys(Keys.PAGE_DOWN)
    rnd = random.random()
    time.sleep(rnd)
    count += img4page

######################
html = browser.page_source
soup = BeautifulSoup(html, "html.parser")
img = soup.findAll("img")
imgs = browser.find_element_by_tag_name("img")
print(imgs)
for im in imgs:
    im.click()
sys.exit()

filenum = 0
srcURL = []
for line in img:
    if str(line).find("src") != -1 and str(line).find("http") < 100:
        srcURL.append(line["src"])
        filenum += 1


for i, src in zip(range(filenum), srcURL):
    urllib.request.urlretrieve(src, os.path.join(target_dir, f"{i}.jpg"))
    print(i, "saved!")

browser.close()

