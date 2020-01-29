import random
import os
import argparse
import shutil
from os import walk
import pickle
from operator import itemgetter
import numpy as np
import time

#### for preprocessing ###
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
import torch
### simple classifiers ###

####### for multi processing #####
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue
import subprocess

def makeCluster(data_directory, vocabHash, kmerSize, wordvecs, k, inq):	
	word_dim = wordvecs.shape[0]
	centroids = None
	while True:
		sample = inq.get()
		if sample is None:
			break
		
		###### go through each sequence and count the number of sequences
		total_sequence = 0
		with open(os.path.join(data_directory, sample), 'r') as f:
			for line in f:
				if line != '' and line != '\n' and line != ' ':
					if line.startswith('>'):
						total_sequence += 1

		### make a minibatch kmeans model ##
		batch_size = 1000
		kmeans = MiniBatchKMeans(n_clusters=k, batch_size = batch_size)

		# #### now go through each fasta file and do kmean ###
		# fasta_matrix = np.zeros((total_sequence, word_dim), dtype = np.float32)
		
		fasta_matrix = np.zeros((batch_size, word_dim), dtype = np.float32)
		with open(os.path.join(data_directory, sample), 'r') as f:
			sequence = ''
			sequence_number = -1
			for line in f:
				line = line.strip()
				if line != '' and line != '\n' and line != ' ':
					if line.startswith('>'):
						sequence_number += 1
						## have we reached the batch size. If yes we cluster
						if sequence_number >= batch_size:
							fasta_matrix = preprocessing.normalize(fasta_matrix, norm = 'l2', axis = 1)
							kmeans = kmeans.partial_fit(fasta_matrix)
							fasta_matrix = np.zeros((batch_size, word_dim), dtype = np.float32)
							sequence_number = 0
						## if not then continue parsing string
						if sequence != '':
							sequence = sequence.lower()
							#### parse through the sequence and update the row in fasta_matrix for this seq
							for i in range(len(sequence) - kmerSize + 1):
								try:
									fasta_matrix[sequence_number] += wordvecs[: , vocabHash[sequence[i : i + kmerSize]]]
								except:
									continue
							sequence = ''
					else:
						sequence = sequence + line
		## any remaining?
		if sequence_number >= 0:
			fasta_matrix = fasta_matrix[:sequence_number][:]
			fasta_matrix = preprocessing.normalize(fasta_matrix, norm = 'l2', axis = 1)
			kmeans = kmeans.partial_fit(fasta_matrix)
		fasta_matrix = preprocessing.normalize(fasta_matrix, norm = 'l2', axis = 1)
		kmeans = kmeans.partial_fit(fasta_matrix)
		del fasta_matrix
		## get the cluster centroid
		centroids = kmeans.cluster_centers_
		##### save the sorted centroids ###
		np.save(os.path.join(data_directory, sample.split('.')[0] + '_centroids'), centroids)
		print("\nFinished clustering: " + sample)


def main(vocabHashFile, trainedModelFile, data_directory, kmeans):
	### load the vocabHash file ####
	with open(os.path.join(vocabHashFile), 'rb') as vocabHashFileHandle:
		vocabHash = pickle.load(vocabHashFileHandle)
	vocabSize = len(vocabHash)
	for km in vocabHash.keys():
		k = len(km)
		break
	print("Kmer Size (k) = " + str(k) + " and total words in vocabulary: " + str(vocabSize))

	############## open the data directory and list samples #####
	samples = []
	for (dirpath, dirnames, filenames) in walk(data_directory):
		samples.extend(filenames)
		break
	samples = [s for s in samples if any(w in s for w in [".fa", ".fna", ".fasta"])]
	
	######### open the trainedModelFile and get the trained word vectors ####
	s = torch.load(trainedModelFile, map_location = 'cpu')
	wordvecs = s['state_dict']['words'].data.numpy()

	#### we have vocabHash, sample directories and wordvecs now
	### create multiprocess (5) each with 11 threads (total 55 workers, we have 56 cores)
	### each of these 5 multi processes will be responsible for a sample

	### for parallel processing ###
	inq =  multiprocessing.Queue()
	num_workers = 2
	workers=[]    
	for i in range(num_workers):
		tmp = multiprocessing.Process(target=makeCluster, args=(data_directory, vocabHash, k, wordvecs, kmeans, inq, ))
		tmp.daemon=True
		tmp.start()
		workers.append(tmp)

	### add the samples to inq 
	for s in samples:
		inq.put(s)

	for i in range(num_workers):
		inq.put(None)
	for w in workers:
		w.join()
	print('\nFinished!')




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-vocabHashFile', help = "Specify the file containing vocabulary index (python dict).", dest = "vocabHashFile", required = True)
	parser.add_argument('-trainedModelFile', help = "Specify the file containing the trained model.", dest = "trainedModelFile", required = True)
	parser.add_argument('-data_directory', help = "Specify the data directory.", dest = "data_directory", required = True)
	parser.add_argument('-kmeans', help = "Value of K in kmeans clustering", dest = "kmeans", default = 4096, type = int)
	args = parser.parse_args()
	main(args.vocabHashFile, args.trainedModelFile, args.data_directory, args.kmeans)