from os import listdir
from os.path import isfile, join
mypath = '/Users/nehamotlani/Desktop/College_Courses/Research/funsd_chargrid/chargrid2d-icdar/data/test/images'
testfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mypath = '/Users/nehamotlani/Desktop/College_Courses/Research/funsd_chargrid/chargrid2d-icdar/data/img_temp'
trainfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

file1 = open('/Users/nehamotlani/Desktop/College_Courses/Research/funsd_chargrid/chargrid2d-icdar/data/train_files.txt', 'r')
Lines = file1.readlines()
# print(Lines)
for i in testfiles:
    i = i.split('.')[0]
    # print(i)
    if i in Lines:
        print(i)
        print('F')
for i in testfiles:
    # i = i.split('.')[0]
    # print(i)
    if i in trainfiles:
        print(i)
        print('F')

