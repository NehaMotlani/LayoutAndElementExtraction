from os import listdir
from os.path import isfile, join
import json 

mypath = './labels'
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(filenames)

i = 0

for file in filenames:
	print(file)
	f = open('./labels/'+file) 
	# data = json.load(f)
	json_object = json.load(f)
	for item in json_object:
		item.pop('box')
		item.pop('words')
		item.pop('id')
		item.pop('linking')
		i=i+1
	f.close()
	f = open('./changed/'+file, "w")
	f.write(json.dumps(json_object))
	f.close()	
	print(file,'done')
print("No of entities are",i)

mypath = './changed/'
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
i=0
for file in filenames:
	f = open(mypath+file) 
	print(mypath+file)
	json_object = json.load(f)
	for item in json_object:
		i=i+1
	f.close()
print("changed test data entity are",i)