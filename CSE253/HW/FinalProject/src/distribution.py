import urllib, json
import requests
import csv
import time 
import os
from pandas import DataFrame
import pickle
import numpy as np
import matplotlib.pyplot as plt

def analysis(image_path):
	api_key = "piN66x5WUrRRV2PGrnM7hhJ1f-uzi8Nk"
	api_secret = "c1ZpOApLOxVe6j9Y-IJ7uDnP3iPME1Vs"
	detect_utl = "https://api-us.faceplusplus.com/facepp/v3/detect"

	data1={ "api_key": api_key,"api_secret":api_secret,"return_landmark":0, "return_attributes":"age"}
	print(image_path)	
	files= {"image_file": open(image_path,'rb')}
	try:
		response=requests.post(detect_utl,data=data1,files=files)
		time.sleep(2)
		req_con=response.content.decode('utf-8')
		req_con = json.loads(req_con)		
		age = req_con['faces'][0]['attributes']['age']['value']
		print("Analysis Done")
		return age
	except:
		print("No Face")
		return None


def writeExcel(image_list, gender_list, age_list):
	data = {
	"image":image_list,
	"gender":gender_list,
	"age":age_list
	}
	df=DataFrame(data)
	df.to_excel("new.xlsx")
	print("Finish Wirte Excel")

def splitAge():
	filePath = "./list_attr_celeba.txt"
	with open(filePath, "r") as f:
		data = f.readlines()
	data = data[2:]
	ageDict = {}
	trainDict = {'train':[]}
	for item in data:
		temp = item.strip().split(" ")
		temp = [i for i in temp if i!=""]
		filename = temp[0]
		age = temp[-1]
		if age == -1:
			age = 0
		ageDict[filename] = age
		trainDict["train"].append(filename)

	with open("./data/ageLabel_celebA.pickle","wb") as f:
		pickle.dump(ageDict,f )
		
	with open("./data/partition_celebA.pickle","wb") as f:
		pickle.dump(trainDict,f)

def getAge():
	filePath = "./list_attr_celeba.txt"
	with open(filePath, "r") as f:
		data = f.readlines()
	index = 40
	title = data[1].strip().split(" ")
	newtitle = ["filename"]
	newtitle.extend(title)
	print(newtitle)
	print("Title:", newtitle[index])
	data = data[2:]
	ageLabel = []
	positive_samples = []
	negative_samples = []	
	for item in data:					
		temp = item.strip().split(" ")
		temp = [i for i in temp if i!=""]
		filename = temp[0]
		age = int(temp[index])
		if age == -1:			
			age = 0
			negative_samples.append((filename, age))
			ageLabel.append(age)
		else:
			positive_samples.append((filename, age))
			ageLabel.append(age)


	print("neg:",len(negative_samples))
	print("pos:",len(positive_samples))

	shuffle_index = np.random.permutation(len(positive_samples))

	trainLabel = []
	trainLabel2 = []
	trainLabel3 = []
	for item in shuffle_index[:len(negative_samples)]:
		trainLabel.append(positive_samples[item])
	trainLabel.extend(negative_samples)
	print(len(trainLabel))

	for item in shuffle_index[len(negative_samples):len(negative_samples)*2]:
		trainLabel2.append(positive_samples[item])	
	trainLabel2.extend(negative_samples)
	print(len(trainLabel2))

	for item in shuffle_index[len(negative_samples)*2:len(negative_samples)*3]:
		trainLabel3.append(positive_samples[item])	
	trainLabel3.extend(negative_samples)
	print(len(trainLabel3))	




	trainLabel = sorted(trainLabel, key=lambda student: student[0])
	trainLabel2 = sorted(trainLabel2, key=lambda student: student[0])
	trainLabel3 = sorted(trainLabel3, key=lambda student: student[0])

	ori_path = "./data/resized_celebA/celebA/"
	des_path  = "./celebAWrapper_1/celebA/"
	des_path2 = "./celebAWrapper_2/celebA/"
	des_path3 = "./celebAWrapper_3/celebA/"

	# print(trainLabel)
	# print(trainLabel2)
	# print(trainLabel3)
	labelFile = []
	count = 0
	for i in trainLabel:
		count += 1
		img = plt.imread(ori_path + i[0])
		plt.imsave(des_path + i[0], arr=img)
		labelFile.append(i[1])
		if (count % 1000) == 0:
			print('%d images complete' % count)
	labelFile = np.array(labelFile)
	with open("./data/ageBalance.pickle","wb") as f:
		pickle.dump(labelFile,f )
	print("==================")
	labelFile2 = []
	count2 = 0
	for i in trainLabel2:
		count2 += 1
		img = plt.imread(ori_path + i[0])
		plt.imsave(des_path2 + i[0], arr=img)
		labelFile2.append(i[1])
		if (count2 % 1000) == 0:
			print('%d images complete' % count2)
	labelFile2 = np.array(labelFile2)
	with open("./data/ageBalance2.pickle","wb") as f:
		pickle.dump(labelFile2,f )
	print("==================")
	labelFile3 = []
	count3 = 0
	for i in trainLabel3:
		count3 += 1
		img = plt.imread(ori_path + i[0])
		plt.imsave(des_path3 + i[0], arr=img)
		labelFile3.append(i[1])
		if (count3 % 1000) == 0:
			print('%d images complete' % count3)
	labelFile3 = np.array(labelFile3)
	with open("./data/ageBalance3.pickle","wb") as f:
		pickle.dump(labelFile3,f )
	
if __name__=="__main__":
	getAge()
		

	
	