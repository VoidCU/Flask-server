
import requests

files = {'files': open('a.jpg','rb')}
eval = requests.post("http://127.0.0.1:5000", files = files)
print(eval.text)