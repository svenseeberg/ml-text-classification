#!/usr/bin/env python3
import requests
import sys
import json

data = [
    {'category': 'Living', 'text': 'Someone is living in a house.'},
    {'category': 'Documentation', 'text': 'Bob works for Documentation.'},
    {'category': 'Documentation', 'text': 'Alice is writing documents.'},
    {'category': 'Linux', 'text': 'SUSE is an Open Source Linux company.'},
    {'category': 'Linux', 'text': 'openSUSE is based on the Linux kernel.'},
]

with open('training-clean.json') as f:
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
