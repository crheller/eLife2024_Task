import requests

# url = "https://datadryad.org/api/v2/datasets/doi%10.5061%2Fdryad.3bk3j9knq/download"
# url = "http://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.3bk3j9knq" # works?
url = "http://datadryad.org/stash/dataset/doi%10.6076%252Fdryad.D1TP4S" # works, but just gives html of webpage
url = "http://datadryad.org/stash/downloads/file_stream/2619507" # fails with permission error

response = requests.get(url)

with open("test.xlsx", mode="wb") as file:
     file.write(response.content)