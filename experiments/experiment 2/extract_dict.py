import os

D = {}
PATH = os.path.join('dataset_1','class_to_id.csv')
with open(PATH,'r') as f:
	for line in f:
		LABEL,ID = line.split(',')
		D[int(ID)] = LABEL

print(D)

