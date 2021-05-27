# -*- coding: utf-8 -*-
import json
mypath = '/Users/nehamotlani/Desktop/College_Courses/Research/Code/bertword_chargrid/chargrid2d-icdar/data/'
with open(mypath+'char2idx.json', encoding='utf-8-sig') as f:
    corpus = json.load(f)
with open(mypath+'embeddings.json', encoding='utf-8-sig') as f:
    embeddings = json.load(f)

# Iterating through the json
# list
dictt = {}
for i in corpus:
	if(i in embeddings):
		dictt[corpus[i]] = embeddings[i]
	else:
		dictt[corpus[i]] = [0]*768
with open("/Users/nehamotlani/Desktop/College_Courses/Research/Code/bertword_chargrid/chargrid2d-icdar/data/final_embeddings.json", "w") as outfile: 
    json.dump(dictt, outfile)