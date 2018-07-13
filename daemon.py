#!/usr/bin/env python3
import os
import cherrypy
import simplejson
import numpy as np
import nltk
from nltk.stem import *
import json

debug = False

config = {
	'global' : {
		'server.socket_host' : '127.0.0.1',
		'server.socket_port' : 8082,
		'server.thread_pool' : 8,
	}
}

class restApi(object):
    def __init__(self, textClassifier):
        self.textClassifier = textClassifier

    @cherrypy.expose
    def add(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))
        body = simplejson.loads(rawbody)
        self.textClassifier.addTrainingSet(body)
        return "1"

    @cherrypy.expose
    def train(self):
        self.textClassifier.train()
        return "1"

    @cherrypy.expose
    def classify(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))
        body = simplejson.loads(rawbody)
        self.textClassifier.classify(body[0])
        return "1"

    @cherrypy.expose
    def flush(self):
        self.textClassifier.flush()
        return "1"


    @cherrypy.expose
    def exit(self):
        for i in range(mp.cpu_count()):
            self.itemQueue.put(None)
        cherrypy.engine.exit()

    @cherrypy.expose
    def index(self):
        return "404"


class TextClassifier(object):
    def __init__(self):
        #self.stemmer = lancaster.LancasterStemmer()
        self.stemmer = porter.PorterStemmer()
        self.words = []
        self.ignore_words = ['?', '.', ',', '\'', '"']
        self.trainingData = []
        self.trainingDocuments = []
        self.categories = []
        self.trainingInputs = []
        self.trainingOutputs = []
        self.errorThreshold = 0.2
        self.synapseFile = "synapses.json"
        self.synapse_0 = None
        self.synapse_1 = None
        self.loadSynapses()

    def flush(self):
        self.words = []
        self.trainingData = []
        self.trainingDocuments = []
        self.categories = []
        self.trainingInputs = []
        self.trainingOutputs = []

    def addTrainingSet(self, trainingSet):
        self.updateWords(trainingSet)
        self.generateTrainingVectors()
        if(debug):
            print("==== RESULTS ====")
            print(self.words)
            print(self.trainingInputs)
            print(self.trainingOutputs)
            print(self.trainingData)

    def updateWords(self, trainingSet):
        self.trainingData.append(trainingSet)
        tokenized_text = nltk.word_tokenize(trainingSet['text'])
        words = [self.stemmer.stem(w.lower()) for w in tokenized_text if w not in self.ignore_words]
        self.words.extend(words)
        self.words = list(set(self.words))
        self.categories.append(trainingSet['category'].lower())
        self.categories = list(set(self.categories))
        self.trainingDocuments.append((trainingSet['category'].lower(), words))

    def generateTrainingVectors(self):
        self.trainingInputs = []
        self.trainingOutputs = []
        for category, words in self.trainingDocuments:
            vecInput = []
            for w in self.words:
                vecInput.append(1) if w in words else vecInput.append(0)
            self.trainingInputs.append(vecInput)
            vecOutput = []
            for c in self.categories:
                vecOutput.append(1) if c == category else vecOutput.append(0)
            self.trainingOutputs.append(vecOutput)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoidDerivative(self, x):
        return x*(1-x)
    
    def cleanSentence(self, text):
        words = nltk.word_tokenize(text)
        words = [self.stemmer.stem(word.lower()) for word in words]
        return words

    def bow(self, sentence):
        sentence_words = self.cleanSentence(sentence)
        bag = [0]*len(self.words)  
        for s in sentence_words:
            for i,w in enumerate(self.words):
                if w == s: 
                    bag[i] = 1
        return(np.array(bag))

    def think(self, sentence):
        print(sentence)
        x = self.bow(sentence.lower())
        l0 = x
        l1 = self.sigmoid(np.dot(l0, self.synapse_0))
        l2 = self.sigmoid(np.dot(l1, self.synapse_1))
        return l2

    def train(self, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
        if(len(self.trainingDocuments) == 0):
            print("No training data.")
            return False
        X = np.array(self.trainingInputs)
        y = np.array(self.trainingOutputs)
        print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
        print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(self.categories)) )
        np.random.seed(1)
        last_mean_error = 1
        self.synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
        self.synapse_1 = 2*np.random.random((hidden_neurons, len(self.categories))) - 1
        prev_synapse_0_weight_update = np.zeros_like(self.synapse_0)
        prev_synapse_1_weight_update = np.zeros_like(self.synapse_1)
        synapse_0_direction_count = np.zeros_like(self.synapse_0)
        synapse_1_direction_count = np.zeros_like(self.synapse_1)
        for j in iter(range(epochs+1)):
            layer_0 = X
            layer_1 = self.sigmoid(np.dot(layer_0, self.synapse_0))
            if(dropout):
                layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))
            layer_2 = self.sigmoid(np.dot(layer_1, self.synapse_1))
            layer_2_error = y - layer_2
            if (j% 10000) == 0 and j > 5000:
                if np.mean(np.abs(layer_2_error)) < last_mean_error:
                    print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                    last_mean_error = np.mean(np.abs(layer_2_error))
                else:
                    print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                    break
            layer_2_delta = layer_2_error * self.sigmoidDerivative(layer_2)
            layer_1_error = layer_2_delta.dot(self.synapse_1.T)
            layer_1_delta = layer_1_error * self.sigmoidDerivative(layer_1)
            synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
            if(j > 0):
                synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
                synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
            self.synapse_1 += alpha * synapse_1_weight_update
            self.synapse_0 += alpha * synapse_0_weight_update
            prev_synapse_0_weight_update = synapse_0_weight_update
            prev_synapse_1_weight_update = synapse_1_weight_update
        synapse = {'synapse0': self.synapse_0.tolist(), 'synapse1': self.synapse_1.tolist(),
                'words': self.words,
                'categories': self.categories
                }
        with open(self.synapseFile, 'w') as outfile:
            json.dump(synapse, outfile, indent=4, sort_keys=True)

    def loadSynapses(self):
        with open(self.synapseFile) as data_file: 
            synapse = json.load(data_file) 
            self.synapse_0 = np.asarray(synapse['synapse0']) 
            self.synapse_1 = np.asarray(synapse['synapse1'])

    def classify(self, sentence):
        if(self.synapse_0 is None):
            print("No trained network yet.")
            return False
        results = self.think(sentence)

        results = [[i,r] for i,r in enumerate(results) if r>self.errorThreshold ] 
        results.sort(key=lambda x: x[1], reverse=True) 
        return_results =[[self.categories[r[0]],r[1]] for r in results]
        print ("%s \n classification: %s" % (sentence, return_results))
        return return_results

if __name__ == '__main__':
    textClassifier = TextClassifier()
    cherrypy.quickstart(restApi(textClassifier),'/', config)
