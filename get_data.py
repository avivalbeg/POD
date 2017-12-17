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

# URL of the image to be downloaded is defined as image_url
for elem in links[1:]:
    r = requests.get('https://web.archive.org/web/20051029232106/http://games.cs.ualberta.ca:80/poker/IRC/IRCdata/'+elem) # create HTTP response object

    # send a HTTP request to the server and save
    # the HTTP response in a response object called r
    with open(elem,'wb') as f:

    # Saving received content as a png file in
    # binary format

    # write the contents of the response (r.content)
    # to a new file in binary mode.
        f.write(r.content)
        f.close()
