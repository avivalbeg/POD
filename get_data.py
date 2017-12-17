from bs4 import BeautifulSoup
import requests 

def find_files(url):

    soup = BeautifulSoup(requests.get(url).text)

    for a in soup.find_all('a'):
        yield a['href']

url = 'https://web.archive.org/web/20051029232106/http://games.cs.ualberta.ca:80/poker/IRC/IRCdata/'

links = []
for link in find_files(url):
    links.append(link)
    
links = links[4:]
image_url = links[0]
for elem in links[1:]:
    r = requests.get('https://web.archive.org/web/20051029232106/http://games.cs.ualberta.ca:80/poker/IRC/IRCdata/'+elem) # create HTTP response object
    with open(elem,'wb') as f:
        f.write(r.content)
        f.close()
