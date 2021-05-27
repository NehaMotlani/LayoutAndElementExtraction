import torch
import json
import numpy as np
# Opening JSON file
f = open('/Users/nehamotlani/Desktop/College_Courses/Research/Code/bertword_chargrid/chargrid2d-icdar/data/final_embeddings.json',)
  
# returns JSON object as 
# a dictionary
data = json.load(f)
class OneHotEncoder():
	def __init__(self, corpus):
		self.corpus = corpus
		self.classes = len(self.corpus) + 1

	def process(self, input):
		# one_hot = torch.FloatTensor(768, input.size(1), input.size(2)).zero_()
		one_hot = np.empty((512, 512, 768))
		# print
		# target = one_hot.scatter_(0, input.data, 1)
		# print(target.size())
		# return target
		# print(input)
		for j in range(input.size(1)):
			for k in range(input.size(2)):
				if(str(int(input[0][j][k])) not in data):
					print(str(int(input[0][j][k])))
					temp = np.zeros(768,dtype=float)
				else:
					temp = data[str(int(input[0][j][k]))]
				one_hot[j][k] = temp
		# print(one_hot.size(0),one_hot.size(1),one_hot.size(2))
		b = np.transpose(one_hot, (2, 0, 1))
		b = torch.FloatTensor(b)
		return b