# Description
This is a lightweight REST daemon that does text classification based on machine learning. The code is based on https://iamtrask.github.io/2015/07/12/basic-python-network/ and https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6. It works only for smaller amounts of text.

This is a fun project that was created during SUSE Hackweek 17.

# Installation
1. ```git clone git@github.com:svenseeberg/ml-text-classification.git```
2. ```cd ml-text-classification```
3. ```python3 -m venv .env```
4. ```source .env```
5. ```pip3 install nltk requests CherryPy simplejson```

# Run Demo
1. ```python3 daemon.py```
2. Start another terminal
3. ```python3 send.py add```
4. ```python3 send.py train```
5. ```python3 send.py classify```

Wait for REST requests to finish before sending the next. While CherryPy supports multithreading, the ClassifyText class is not thread safe.
