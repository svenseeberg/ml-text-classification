#!/usr/bin/env python3
import requests
import sys
import json

with open('training.json') as f:
    data = json.load(f)

if(len(sys.argv) == 1 or sys.argv[1] == "add"):
    x = 0
    y = len(data)
    for item in data:
        r = requests.post('http://localhost:8082/add/', json=item)
        x = x + 1
        print(str(x)+" of "+str(y))
elif(sys.argv[1] == "train"):
    r = requests.post('http://localhost:8082/train/')
elif(sys.argv[1] == "flush"):
    r = requests.post('http://localhost:8082/flush/')
elif(sys.argv[1] == "train"):
    r = requests.post('http://localhost:8082/train/')
elif(sys.argv[1] == "classify"):
    item = ['Charlie is reading documentation.']
    r = requests.post('http://localhost:8082/classify/', json=item)
