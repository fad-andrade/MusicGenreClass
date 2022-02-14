import os
import random
import shutil

train_path = './data/train'
test_path = './data/test'

test_percentage = 0.2

if(os.path.exists(test_path)):
	shutil.rmtree(test_path)
os.makedirs(test_path)

for genre in os.listdir(train_path):
	os.makedirs(f'{test_path}/{genre}')	
	
	files_list = os.listdir(f'{train_path}/{genre}')
	random.shuffle(files_list)
	
	split_number = int(test_percentage * len(files_list))
	files_list = files_list[0 : split_number]
	
	for file in files_list:
		shutil.move(f'{train_path}/{genre}/{file}', f'{test_path}/{genre}/{file}')
