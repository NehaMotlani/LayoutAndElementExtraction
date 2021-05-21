import json
from os import listdir
from os.path import isfile, join
root = '/Users/nehamotlani/Desktop/College_Courses/Research/Code/basecode/chargrid2d-icdar/'
def EditDistDP(str1, str2):
	len1 = len(str1)
	len2 = len(str2)
	DP = [[0 for i in range(len1 + 1)] 
			 for j in range(2)];
	for i in range(0, len1 + 1):
		DP[0][i] = i

	for i in range(1, len2 + 1): 
		for j in range(0, len1 + 1):
			if (j == 0):
				DP[i % 2][j] = i
			elif(str1[j - 1] == str2[i-1]):
				DP[i % 2][j] = DP[(i - 1) % 2][j - 1] 
			else:
				DP[i % 2][j] = (1 + min(DP[(i - 1) % 2][j], min(DP[i % 2][j - 1], DP[(i - 1) % 2][j - 1])))       
	return DP[len2 % 2][len1]

groundtruthfiles = [f for f in listdir(root+'data/test/changed') if isfile(join(root+'data/test/changed', f))]
predictedfiles = [f for f in listdir(root+ 'data/output') if isfile(join(root+'data/output', f))]

n = 0
n0 = 0
n1 = 0
n2 = 0
n3 = 0
for file in groundtruthfiles:
	groundtruth = open(root+'data/test/changed/'+file)
	predicted = open(root+'data/output/'+file)
	data_gt = json.load(groundtruth)
	data_p = json.load(predicted)
	class0_gt = []
	class1_gt = []
	class2_gt = []
	class3_gt = []
	class0_p = []
	class1_p = []
	class2_p = []
	class3_p = []
	for obj in data_gt:
		if obj['class'] == 'other':
			class0_gt.append(obj['text'])
		elif obj['class'] == 'question':
			class1_gt.append(obj['text'])
		elif obj['class'] == 'header':
			class2_gt.append(obj['text'])
		elif obj['class'] == 'answer':
			class3_gt.append(obj['text'])
	
	for obj in data_p:
		if obj['class'] == 'other':
			class0_p.append(obj['text'])
		elif obj['class'] == 'question':
			class1_p.append(obj['text'])
		elif obj['class'] == 'header':
			class2_p.append(obj['text'])
		elif obj['class'] == 'answer':
			class3_p.append(obj['text'])
	# print(class0_gt,class1_gt,class2_gt,class3_gt)
	n += len(class0_gt)+len(class1_gt)+len(class2_gt)+len(class3_gt)	
	n0 += len(class0_gt)
	n1 += len(class1_gt)
	n2 += len(class2_gt)
	n3 += len(class3_gt)

	class0_gt.sort()
	class1_gt.sort()
	class2_gt.sort()
	class3_gt.sort()

	class0_p.sort()
	class1_p.sort()
	class2_p.sort()
	class3_p.sort()

	stringclass0_gt = (" ").join(class0_gt)
	stringclass1_gt = (" ").join(class1_gt)
	stringclass2_gt = (" ").join(class2_gt)
	stringclass3_gt = (" ").join(class3_gt)

	stringclass0_p = (" ").join(class0_p)
	stringclass1_p = (" ").join(class1_p)
	stringclass2_p = (" ").join(class2_p)
	stringclass3_p = (" ").join(class3_p)

	ans0 = EditDistDP(stringclass0_p,stringclass0_gt)
	ans1 = EditDistDP(stringclass1_p,stringclass1_gt)
	ans2 = EditDistDP(stringclass2_p,stringclass2_gt)
	ans3 = EditDistDP(stringclass3_p,stringclass3_gt)

print('overall ',1-((ans0+ans1+ans2+ans3)/n))
print('other ',1-(ans0/n0))
print('question ',1-(ans1/n1))
print('header ',1-(ans2/n2))
print('answer ',1-(ans3/n3))

y_pred = []
y_actual = []
groundtruth_entities = 0
# 0->other 1->question 2->header 3->answer 
for file in groundtruthfiles:
	if(file=='.DS_Store'):
		print('here')
		continue
	groundtruth = open(root+'data/test/changed/'+file)
	predicted = open(root+'data/output/'+file)
	data_gt = json.load(groundtruth)
	data_p = json.load(predicted)
	dict_data_gt = {}
	dict_data_p = {}
	for obj in data_gt:
		dict_data_gt[obj['text']] = obj['class']
	for obj in data_p:
		dict_data_p[obj['text']] = obj['class']
	for i in dict_data_gt:
		groundtruth_entities += 1
		if(dict_data_gt[i]=='other'):
			y_actual.append('other')
		elif(dict_data_gt[i]=='question'):
			y_actual.append('question')
		elif(dict_data_gt[i]=='header'):
			y_actual.append('header')
		else:
			y_actual.append('answer')
		
		if(dict_data_p[i]=='other'):
			y_pred.append('other')
		elif(dict_data_p[i]=='question'):
			y_pred.append('question')
		elif(dict_data_p[i]=='header'):
			y_pred.append('header')
		else:
			y_pred.append('answer')	
print(groundtruth_entities)
from sklearn import metrics
print(metrics.confusion_matrix(y_actual, y_pred))

# Print the precision and recall, among other metrics
print(metrics.classification_report(y_actual, y_pred, digits=3))
