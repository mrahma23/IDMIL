import random
import os
import argparse
import shutil
from os import walk
import pickle
from operator import itemgetter
import math
import numpy as np
import time
import scipy.stats as st

#### for the trained model ##
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from cnn import CNN_net
#### for preprocessing ###
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


def convertToImage_colwise(v, device = 'cpu'):
	d = []
	size = (int)(math.sqrt(v.shape[0])) 
	for i in range(v.shape[1]):
		cur_column = v[:, i]
		d.append(cur_column)
	d = np.stack(d)
	d = torch.FloatTensor(d)
	return d


def standardScaler(data_directory, selected_samples):
	scaler = preprocessing.StandardScaler()
	for s in selected_samples:
		p = os.path.join(data_directory, s)
		v = np.load(p)
		scaler.partial_fit(v)
	return scaler


def makeSum(data_directory, selected_samples, scaler):
	p = os.path.join(data_directory, selected_samples[0])
	v = np.load(p)
	v = scaler.transform(v)
	t = np.sum(v, axis = 0)
	for s in selected_samples[1:]:
		p = os.path.join(data_directory, s)
		v = np.load(p)
		v = scaler.transform(v)
		t += np.sum(v, axis = 0)
	t = t.reshape(1,-1)
	return t


def rearrange(href, centroids):
	#### calculate distance from href and sort the indices ###
	distances = pairwise_distances(X = centroids, Y = href, metric = 'cosine')
	distances = distances.reshape(1,-1)
	sorted_indices = np.argsort(distances)
	sorted_indices = sorted_indices.reshape(1,-1)
	sorted_indices = sorted_indices[0].tolist()
	##### arrange the cluster matrix based on sorted indices w.r.t the distances from href ####
	sorted_centroids = np.zeros((centroids.shape[0], centroids.shape[1]), dtype = np.float32)
	for i in range(centroids.shape[0]):
		sorted_centroids[i, : ] = centroids[sorted_indices[i], : ]
	return sorted_centroids


def shuffle_matrix(a, min_window, max_window):
	row_size = a.shape[0]
	row_indices = [i for i in range(row_size)]
	i = 0
	while True:
		w = np.random.randint(min_window,max_window)
		if i + w >= row_size:
			break
		copy = row_indices[i:i+w]
		random.shuffle(copy)
		row_indices[i:i+w] = copy
		i += w
	a = a[row_indices]
	return a


def getDataset(train_directory, test_directory, class_to_samples_map, sample_to_class_map, repeats, device = 'cpu'):	
	####### Go to each directory, list all files and prepare inq items ####
	train_set = []
	test_set = []
	train_labels = []
	test_labels = []

	### samples in train directory 
	tr_samples = []
	for (dirpath, dirnames, filenames) in walk(train_directory):
		tr_samples.extend(filenames)
		break
	tr_samples = [s for s in tr_samples if ".npy" in s]

	### samples in test directory 
	tst_samples = []
	for (dirpath, dirnames, filenames) in walk(test_directory):
		tst_samples.extend(filenames)
		break
	tst_samples = [s for s in tst_samples if ".npy" in s]

	### prepare scaler and sum on training set
	scaler = standardScaler(train_directory, tr_samples)
	t_sum = makeSum(train_directory, tr_samples, scaler)
	
	#### load the train vectors, add them to the train set ###
	for s in tr_samples:
		p = os.path.join(train_directory, s)
		v = np.load(p)
		v = scaler.transform(v)
		v = rearrange(t_sum, v)
		v_img = convertToImage_colwise(v, device)
		train_set.append(v_img)
		sample = s.split('.')[0].split('_')[0]
		train_labels.append(sample_to_class_map[sample])

		####### shuffle and repeat #####
		if repeats > 0:
			for r in range(repeats):
				v_shuffled = shuffle_matrix(v, 1, 10)
				v_img = convertToImage_colwise(v_shuffled, device)
				train_set.append(v_img)
				train_labels.append(sample_to_class_map[sample])
	
	#### load the test vectors, add them to the test set ###
	for s in tst_samples:
		p = os.path.join(test_directory, s)
		v = np.load(p)
		v = scaler.transform(v)
		v = rearrange(t_sum, v)
		v_img = convertToImage_colwise(v, device)
		test_set.append(v_img)
		sample = s.split('.')[0].split('_')[0]
		test_labels.append(sample_to_class_map[sample])
	
	print("Augmented training set size: " + str(len(train_set)) + "; Test set size: " + str(len(test_set)))
	return train_set, test_set, np.array(train_labels), np.array(test_labels)


def predict(train_directory, test_directory, class_to_samples_map, sample_to_class_map, wordvecs, learning_rate, batch_size, epoch, repeats, device = 'cpu'):
	train_set, test_set, train_labels, test_labels = getDataset(train_directory, test_directory, class_to_samples_map, sample_to_class_map, repeats, device)

	permutation = [ind for ind in range(len(train_set))]
	random.shuffle(permutation)
	batch_size = min(batch_size, len(train_set))

	net = CNN_net().to(device)
	#optimizer = optim.ADAM(net.parameters(), lr = learning_rate)
	optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=0.9)
	criterion = nn.NLLLoss()
	start = time.time()
	train_labels = torch.LongTensor(train_labels).to(device)
	for e in range(epoch):
		l = 0
		for i in range(0, len(train_set), batch_size):
			indices = permutation[i:i+batch_size]
			batch_x = []
			batch_y = []
			for ind in indices:
				batch_x.append(train_set[ind])
			batch_x = torch.stack(batch_x)
			batch_y = train_labels[indices]
			data, target = Variable(batch_x).to(device), Variable(batch_y).to(device)
			
			optimizer.zero_grad()
			net_out = net(data)
			loss = criterion(net_out, target)
			l += loss.data
			loss.backward()
			optimizer.step()
		print("\tepoch: " + str(e+1) + " with avg loss: " + str((float)(l)/batch_size))
	
	test_set = torch.stack(test_set).to(device)
	net_out = net(test_set)
	predicted_test_labels = net_out.data.max(1)[1].to('cpu').numpy()
	acc = accuracy_score(test_labels, predicted_test_labels)
	duration = time.time() - start
	return acc, duration


def main(vocabHashFile, trainedModelFile, train_directory, test_directory, sample_class_info, learning_rate, batch_size, epoch, trial, repeats, device):
	### load the vocabHash file ####
	with open(os.path.join(vocabHashFile), 'rb') as vocabHashFileHandle:
		vocabHash = pickle.load(vocabHashFileHandle)
	vocabSize = len(vocabHash)
	for km in vocabHash.keys():
		k = len(km)
		break
	print("\nKmer Size (k) = " + str(k) + " and total words in vocabulary: " + str(vocabSize))

	############## open the sample_class_info and get the semple directories #####
	class_to_samples_map = {}
	sample_to_class_map = {}
	with open(os.path.join(sample_class_info)) as f:
		for line in f:
			if line != "" and line != " " and line != "\n":
				line = line.strip().split('\t')
				sample = line[0].split('.')[0]
				class_label = (int)(line[1])
				
				try:
					class_to_samples_map[class_label].append(sample)
				except:
					class_to_samples_map[class_label] = [sample]
				
				try:
					c = sample_to_class_map[sample]
				except:
					sample_to_class_map[sample] = class_label


	######### open the trainedModelFile and get the trained word vectors ####
	s = torch.load(trainedModelFile, map_location = 'cpu')
	wordvecs = s['state_dict']['words'].data.numpy()
	
	########### start predicting ######
	accuracies = []
	durations = []
	for i in range(trial):
		print("\n\nStarting Trial: " + str(i+1))
		acc, dur = predict(train_directory, test_directory, class_to_samples_map, sample_to_class_map, wordvecs, learning_rate, batch_size, epoch, repeats, device)
		accuracies.append(acc)
		durations.append(dur)
		print("\tTrial # " + str(i) + "\taccuracy: " + "{:.4f}".format(acc) + "\tduration: " + "{:.4f}".format(dur) + " seconds")

	mean_acc = np.mean(accuracies)
	mean_dur = np.mean(durations)
	interval = st.t.interval(0.95, len(accuracies)-1, loc=np.mean(accuracies), scale=st.sem(accuracies))
	print("Mean accuracy value: " +"{:.4f}".format(mean_acc) + "\tConfidence interval (95%): [" + "{:.4f}".format(interval[0]) + " , " + "{:.4f}".format(interval[1]) + "]\tMean duration: " + "{:.4f}".format(mean_dur) + " seconds")

	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-vocabHashFile', help = "Specify the file containing vocabulary index (python dict).", dest = "vocabHashFile", required = True)
	parser.add_argument('-trainedModelFile', help = "Specify the file containing the trained model.", dest = "trainedModelFile", required = True)
	parser.add_argument('-train_dir', help = "Specify the training data directory.", dest = "train_directory", required = True)
	parser.add_argument('-test_dir', help = "Specify the test data directory.", dest = "test_directory", required = True)
	parser.add_argument('-class_info', help = "Specifiy the file containing sample class info.", dest = "sample_class_info", default = "sample_class_info")
	parser.add_argument('-learning_rate', help = "Specifiy the Learning Rate.", dest = "learning_rate", type = float, default = 0.003)
	parser.add_argument('-batch_size', help = "Specifiy the Batch Size.", dest = "batch_size", type = int, default = 32)
	parser.add_argument('-epoch', help = "Specifiy the total number of Iterations.", dest = "epoch", type = int, default = 100)
	parser.add_argument('-trial', help = "Specifiy the total number of prediction trials.", dest = "trial", type = int, default = 10)
	parser.add_argument('-repeats', help = "Specifiy the total number of repeats in training samples.", type = int, dest = "repeats", default = 5)
	parser.add_argument('-device', help = "Specifiy the device for training: cpu/cuda:0/cuda:1/cuda:2/cuda:3", dest = "device", default = "cpu")
	args = parser.parse_args()
	main(args.vocabHashFile, args.trainedModelFile, args.train_directory, args.test_directory, args.sample_class_info, args.learning_rate, args.batch_size, args.epoch, args.trial, args.repeats, args.device)