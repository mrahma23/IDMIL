################### for analytics #################################################
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import math
import random
import tables as tb
################### for os operations #################################################
import argparse
import sys
import os
from os import walk
from os import listdir
from os.path import isfile, join
import time
import datetime
import pickle
from operator import itemgetter
import gc
import shutil
################## for multiprocessing ##################################################
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue


########## vocab node ############
class VocabWord:
	def __init__(self, word, count = 0):
		self.word = word
		self.count = count
		self.path = None
		self.code = None
	def __eq__(self, other):
		return self.count == other.count
	def __ne__(self, other):
		return not self.count == other.count
	def __lt__(self, other):
		return self.count < other.count
	def __gt__(self, other):
		return self.count > other.count
	def __le__(self, other):
		return self.count <= other.count
	def __ge__(self, other):
		return self.count >= other.count
		

######### the vocabulary class that creates vocabulary and huffman tree for hierarchical softmax ######
class Vocabulary:
	def __init__(self, file, prefix, k, tf_idf_cutoff, useHdf):
		######### following class variables handles files #########
		self.prefix = prefix
		self.file = file
		######### following class variables are specific to vocabulary building ####
		self.k = k
		self.tf_idf_cutoff = tf_idf_cutoff
		self.useHdf = useHdf
		######### following class vaiables are internal representations ######
		self.vocabHash = None
		self.colToVocab = None
		self.vocabCounts = None
		##### the followings are for csr format data for storing words, their paths in binary tree and binary code ###
		self.words_indptr = None
		self.path_indices = None
		self.binary_data = None
		########### following class variables are based on input files and what we have found in the input so far ###
		self.vocabSize = 0
		self.total_num_sequence = 0
		
		########## build the vocabulary #####
		self.vocabularyBuilder()		
		########### Create Huffman codes ########
		self.encode_huffman()
	

	######### reads sequences from the file. This function is a multiprocess and runs parallely in background (daemon) ########## 
	def reader(self, inq, outq):
		allowed = ['a', 'c', 'g', 't']
		word_to_col = {}
		total_kmer_found = -1
		memory_chunk = 50000
		sequence_number = 0
		for inp_f in self.file:
			f = open(inp_f, 'r')
			sequence = ''
			for line in f:
				line = line.strip()
				if line != '' and line != '\n' and line != ' ':
					if line.startswith('>'):
						sequence_number += 1
						if sequence != '':
							######## create the array #########
							memory_chunk = len(sequence) - self.k + 1
							col_indices = np.zeros(memory_chunk, dtype = np.uint32)
							cur_col_ind = 0
							########## add the col numbers ####
							for i in range(len(sequence) - self.k + 1):
								kmer = sequence[i : i + self.k]
								##### find the column number for this kmer in the sparse matrix ######
								col_number = -1
								try:
									col_number = word_to_col[kmer]
								except:
									total_kmer_found += 1
									col_number = total_kmer_found
									word_to_col[kmer] = col_number
								if col_number == -1:
									continue
								col_indices[cur_col_ind] = col_number
								cur_col_ind += 1
								###### check if we need to resize the array ##########
								if cur_col_ind >= len(col_indices) - 1:
									col_indices = np.concatenate((col_indices, np.zeros(memory_chunk, dtype = np.uint32)), axis = None)
							#### trim the array to get unused spaces ####
							if len(col_indices) - 1 >= cur_col_ind:
								col_indices = np.delete(col_indices, np.s_[cur_col_ind:]) 
							##### check the queue. if it is too loaded then wait ###
							while inq.qsize() > 5:
								pass
							inq.put(col_indices)
							##### free up space ####
							del sequence
							sequence = ''
							gc.collect()
							######### at this point we are done with ONE sequence ####
					else:
						line = line.lower()
						trimmed_line = ""
						for ch in line:
							if ch in allowed:
								trimmed_line = trimmed_line + ch
						sequence = sequence + trimmed_line
						del line
			if sequence != '':
				######## create the array #########
				memory_chunk = len(sequence) - self.k + 1
				col_indices = np.zeros(memory_chunk, dtype = np.uint32)
				cur_col_ind = 0
				########## add the col numbers ####
				for i in range(len(sequence) - self.k + 1):
					kmer = sequence[i : i + self.k]
					##### find the column number for this kmer in the sparse matrix ######
					col_number = -1
					try:
						col_number = word_to_col[kmer]
					except:
						total_kmer_found += 1
						col_number = total_kmer_found
						word_to_col[kmer] = col_number
					if col_number == -1:
						continue
					col_indices[cur_col_ind] = col_number
					cur_col_ind += 1
					###### check if we need to resize the array ##########
					if cur_col_ind >= len(col_indices) - 1:
						col_indices = np.concatenate((col_indices, np.zeros(memory_chunk, dtype = np.uint32)), axis = None)
				#### trim the array to get unused spaces ####
				if len(col_indices) - 1 >= cur_col_ind:
					col_indices = np.delete(col_indices, np.s_[cur_col_ind:]) 
				##### check the queue. if it is too loaded then wait ###
				while inq.qsize() > 5:
					pass
				inq.put(col_indices)
				##### free up space ####
				del sequence
				sequence = ''
				gc.collect()
			##### close the current input file ###
			f.close()
		inq.put(None)
		outq.put(word_to_col)

	
	def create_hdf_csr(self, file_name):
		######## create hdf file friom pytables ######
		h5File_path = os.path.join(self.prefix, file_name)
		h5File = tb.open_file(h5File_path, "w")
		filters = tb.Filters(complevel=5, complib='blosc')
		h5File.create_earray(h5File.root, 'data', tb.Float32Atom(), shape=(0,), filters=filters)
		h5File.create_earray(h5File.root, 'indices', tb.UInt32Atom(),shape=(0,), filters=filters)
		h5File.create_earray(h5File.root, 'indptr', tb.UInt32Atom(), shape=(0,), filters=filters)
		h5File.root.indptr.append(np.array([0], dtype = np.uint32))
		return h5File, h5File_path


	def tf_idf_hdf(self, h5File, num_rows, num_cols):
		print("\tComputing tf-tdf scores using hdfs.")
		#### paths and filter ######
		idf_path = os.path.join(self.prefix, "idf.hdf")
		freq_path = os.path.join(self.prefix, "freq.hdf")
		filters = tb.Filters(complevel=5, complib='blosc')
		####### open the hdf file ####
		idf_file = tb.open_file(idf_path, "w")
		freq_file = tb.open_file(freq_path, "w")
		###### create arrays #####
		idf_file.create_carray(idf_file.root, 'idf', tb.Float32Atom(), shape=(num_cols,), filters=filters)
		freq_file.create_carray(freq_file.root, 'freq', tb.Float32Atom(), shape=(num_cols,), filters=filters)
		######## initialize the occurances and idf with zeros ####
		for i in range(num_cols):
			freq_file.root.freq[i] = 0
			idf_file.root.idf[i] = 0
		##### now go through each sequences and calculate idf and total freq ####
		sums = np.zeros((num_rows), dtype = np.float32)
		total = 0
		for i in range(len(h5File.root.indptr) - 1): ####### iterating through sequences
			s = 0
			for j in range(h5File.root.indptr[i], h5File.root.indptr[i+1]):
				idf_file.root.idf[h5File.root.indices[j]] += 1
				freq_file.root.freq[h5File.root.indices[j]] += h5File.root.data[j]
				s += h5File.root.data[j]
			sums[i] = s
			total += s
		#### calculate the tf-idf ########
		for i in range(len(h5File.root.indptr) - 1): ####### iterating through sequences 
			for j in range(h5File.root.indptr[i], h5File.root.indptr[i+1]): ##### iterating through kmers in sequences ###
				h5File.root.data[j] = math.log1p(h5File.root.data[j] / sums[i]) * math.log1p((num_rows / idf_file.root.idf[h5File.root.indices[j]]))
		####### prepare kmer frequencies ####
		for i in range(num_cols):
			freq_file.root.freq[i] = math.log1p(freq_file.root.freq[i] / total)
		##### close the idf file, delete the idf file, delete sums, do garbage collections ####
		idf_file.close()
		try:
			os.remove(idf_path)
		except:
			print("Error deleting temporary file: " + str(idf_path))
			pass		
		return h5File.root.indptr, h5File.root.indices, h5File.root.data, freq_file, freq_path 

	
	def tf_idf_sparseMatrix(self, h5File, num_rows, num_cols):
		print("\tComputing tf-tdf scores using sparse matrix.")
		m = csr_matrix((h5File.root.data[:], h5File.root.indices[:], h5File.root.indptr[:]), shape=(num_rows, num_cols))
		########## calculate idf ###########
		idf_per_word = np.log(1 + (num_rows / m.getnnz(axis = 0)))
		##### calculate occurances of each word (needed for merging in binary tree) #####
		occurance_per_word = m.sum(axis = 0).A.ravel()
		s = sum(occurance_per_word)
		occurance_per_word = np.log(1 + (occurance_per_word / s))
		####### calculate frequecies of the words in each document #####
		f = sparse.diags(1/m.sum(axis = 1).A.ravel())
		m = f @ m
		m = m.log1p()
		gc.collect()
		####### calculate tf-idf #######
		m = m.multiply(idf_per_word).tocsr()
		indptr = m.indptr
		indices = m.indices
		data = m.data
		######### delete items ######
		del m
		del f
		del idf_per_word
		gc.collect()
		return indptr, indices, data, occurance_per_word


	def getTopWords(self, indptr, indices, data):
		print("\tRetrieving top kmers based on tf-idf cutoff.")
		top_words = set()
		for i in range(len(indptr) - 1):
			items = []
			for it in zip(indices[indptr[i]:indptr[i+1]], data[indptr[i]:indptr[i+1]]):
				items.append(it)
			cutoff = len(items)
			####### for testing #####
			if self.tf_idf_cutoff < 1.0:
				cutoff = (int)(cutoff * self.tf_idf_cutoff)
				items.sort(key=itemgetter(1), reverse = True)
				items = items[:cutoff]
			for t in items:
				top_words.add(t[0])
			del items
		return top_words
	

	def vocabularyBuilder(self):
		######## feedbacks to the user ##########
		print("\nBuilding Vocabulary.")
		start = time.time()
		######## create hdf file for pytables ######
		h5File, h5File_path = self.create_hdf_csr("vocabulary_matrix.hdf")
		####### inq is the threadsafe queue ####
		inq = multiprocessing.Queue()
		outq = multiprocessing.Queue()
		##### startreader process #####
		fileReadProcess = multiprocessing.Process(target = self.reader, args = (inq, outq, ))
		fileReadProcess.daemon = True
		fileReadProcess.start()
		######### continuously read from the inq queue the sequence word compositions and make sparse matrix #####
		word_to_col = None
		max_col = -1
		############ gather word counts per sequence ########
		while True:
			item = inq.get()
			if item is None:
				break
			unique, counts = np.unique(item, return_counts=True)
			del item
			unique.astype(np.uint32)
			counts.astype(np.float32)
			max_col = max(max_col, max(unique))
			h5File.root.indices.append(unique)
			h5File.root.data.append(counts)
			cur_val = h5File.root.indptr[-1]
			h5File.root.indptr.append(np.array([cur_val + len(counts)], dtype = np.uint32)) 
		while outq.qsize() == 0:
			pass
		word_to_col = outq.get()
		fileReadProcess.join()
		######## number of rows (sequences) and cols (words) #######
		num_rows = len(h5File.root.indptr) - 1
		num_cols = max_col + 1
		######## now create the sparse matrix #######
		if self.useHdf == False:
			indptr, indices, data, occurance_per_word = self.tf_idf_sparseMatrix(h5File, num_rows, num_cols)
		else:
			indptr, indices, data, freq_file, freq_path = self.tf_idf_hdf(h5File, num_rows, num_cols)
			occurance_per_word = freq_file.root.freq
		########## find top tf-idf words ###########
		top_words =  self.getTopWords(indptr, indices, data)
		self.vocabSize = len(top_words)
		################### Create a table for vocabulary counts and index-to-vocabulary ########################
		vocabCountIndex_path = os.path.join(self.prefix, "vocabCountIndex.hdf")
		filters = tb.Filters(complevel=5, complib='blosc')
		vocabCountIndex_file = tb.open_file(vocabCountIndex_path, "w")
		vocabCountIndex_file.create_carray(vocabCountIndex_file.root, 'count', tb.Float32Atom(), shape=(self.vocabSize,), filters=filters)
		vocabCountIndex_file.create_carray(vocabCountIndex_file.root, 'index', tb.StringAtom(itemsize=self.k), shape=(self.vocabSize,), filters=filters)
		############ Now prune unnecessary vocabularies ##############
		self.vocabHash = {}
		ind = 0
		for kmer, kmer_index in word_to_col.items():
			if kmer_index in top_words:
				self.vocabHash[kmer] = ind
				vocabCountIndex_file.root.index[ind] = str.encode(kmer)
				vocabCountIndex_file.root.count[ind] = occurance_per_word[kmer_index]
				ind += 1
		vocabCountIndex_file.close()
		### save self.vocabHash dictionary #####
		with open(os.path.join(self.prefix, "vocabHash"), 'wb') as outfile:
			pickle.dump(self.vocabHash, outfile, pickle.HIGHEST_PROTOCOL)
		###### delete unnecessary things and force garbage collection ####
		del top_words
		prev_vocabSize = len(word_to_col)
		del word_to_col
		######## close the hdf file and delete it. We dont need it anymore ###
		h5File.close() 
		try:
			os.remove(h5File_path)
			if self.useHdf == True:
				freq_file.close()
				os.remove(freq_path)
		except:
			pass
		######## summarize for the user ##########
		duration = time.time() - start
		print("\tOriginal # of kmers: " + str(prev_vocabSize) + "\n\tPruned # of kmers: " + str(prev_vocabSize - self.vocabSize) + "\n\tNew # of kmers: " + str(self.vocabSize))
		print("Total time to process " + str(num_rows) + " sequences and " + str(num_cols) + " kmers: " + "{:.4f}".format(duration) + \
			" seconds (" + "{:.4f}".format(duration / 60.00) + " minutes).")
  


	def create_parents_binaries(self):
		print("\tCreating Binary tree.")
		count = np.zeros(2 * self.vocabSize - 1, dtype = np.float32)
		vocabCountIndex_path = os.path.join(self.prefix, "vocabCountIndex.hdf")
		vocabCountIndex_file = tb.open_file(vocabCountIndex_path, "r")
		#np.copyto(count[:self.vocabSize], self.vocabCounts)
		for i in range(self.vocabSize):
			count[i] = vocabCountIndex_file.root.count[i]
		vocabCountIndex_file.close()
		for ind in range(self.vocabSize, len(count)):
			count[ind] = 1e15
		parent = np.zeros(2 * self.vocabSize - 2, dtype = np.uint32)
		binary = np.zeros(2 * self.vocabSize - 2, dtype = np.uint32)
		gc.collect()
		pos1 = self.vocabSize - 1
		pos2 = self.vocabSize
		########## create binary tree, parent index (We only need the parent index)
		for i in range(self.vocabSize - 1):
			####### Find minimum 1 ##########
			if pos1 >= 0:
				if count[pos1] < count[pos2]:
					min1 = pos1
					pos1 -= 1
				else:
					min1 = pos2
					pos2 += 1
			else:
				min1 = pos2
				pos2 += 1
			##### Now Find minimum 2 ######
			if pos1 >= 0:
				if count[pos1] < count[pos2]:
					min2 = pos1
					pos1 -= 1
				else:
					min2 = pos2
					pos2 += 1
			else:
				min2 = pos2
				pos2 += 1
			####### assign parent node index ######
			count[self.vocabSize + i] = count[min1] + count[min2]
			parent[min1] = self.vocabSize + i
			parent[min2] = self.vocabSize + i
			binary[min2] = 1
		return parent, binary
	

	def compute_paths_codes(self, parent, binary):
		print("\tComputing node paths and huffman codes for each kmer.")
		root_idx = 2 * self.vocabSize - 2
		c = 0
		for i in range(self.vocabSize):
			path = [] # List of indices from the leaf to the root
			code = [] # Binary Huffman encoding from the leaf to the root
			node_idx = i
			##### find the path #######
			while node_idx < root_idx:
				if node_idx >= self.vocabSize:
					path.append(node_idx - self.vocabSize)
				code.append(binary[node_idx])
				node_idx = parent[node_idx]
			path.append(root_idx - self.vocabSize)
			path = np.array(path, dtype = np.uint32)
			code = np.array(code, dtype = np.uint32)
			yield path, code


	def encode_huffman(self):
		print("\nStarting Huffman encoding.")
		encoding_start_time = time.time()
		parent, binary = self.create_parents_binaries()
		######## create hdf csr sparse matrix to store tree paths and huffman codes #####
		h5File, h5File_path = self.create_hdf_csr("vocabularyPathCodes.hdf")
		for path, code in self.compute_paths_codes(parent, binary):
			h5File.root.indices.append(path)
			h5File.root.data.append(code)
			cur_val = h5File.root.indptr[-1]
			h5File.root.indptr.append(np.array([cur_val + len(code)], dtype = np.uint32)) 
		encoding_dutration = time.time() - encoding_start_time
		print("Total time to encode " + str(self.vocabSize) + " kmers: " + \
			"{:.2f}".format(encoding_dutration) + " seconds (" + "{:.2f}".format(encoding_dutration/60.00) + " minutes).")
		#### close the file. We dont have to store it separately ###
		h5File.close()  
		del binary
		del parent

			
def buildVocab(fileName, filePrefix, k, tf_idf_cutoff, useHdf):
	input_files = []
	########### Some Error checking before starting the program ########
	if os.path.isdir(os.path.join(fileName)) == True:
		for (dirpath, dirnames, filenames) in walk(os.path.join(fileName)):
			input_files.extend(filenames)
		for i in range(len(input_files)):
			input_files[i] = os.path.join(fileName, input_files[i])
	else:
		input_files.append(os.path.join(fileName))
	#random.shuffle(input_files)
	######### check the destination path ######
	filePrefix = os.path.join(filePrefix, "vocabulary")
	if filePrefix != "":
		try:
			shutil.rmtree(os.path.join(filePrefix))
		except:
			pass
		if not os.path.exists(os.path.join(filePrefix)):
			os.makedirs(os.path.join(filePrefix))
	###### check for the value of k ######
	if k <= 0:
		print("Value of k (kmer) should be positive. Setting k to default 8.")
		k = 8
	######### validate tf idf value #########
	if tf_idf_cutoff <= 0:
		print("TF_IDF cutoff value should be positive. Setting TD_IDF cutoff to default 1.00")
		tf_idf_cutoff = 1.0
	############ Parameter list for printing ###########
	print("\nParameters:\n\tTotal # of input files: " + str(len(input_files)) + "\n\tOutput Destination: " + str(filePrefix) + "\n\tk: " +  str(k) +\
	 "\n\tTF-IDF cutoff: " + str(tf_idf_cutoff) + "\n\tUse HDF: " + str(useHdf))
	################ Build Vocabulary ##################
	vb = Vocabulary(input_files, filePrefix, k, tf_idf_cutoff, useHdf)  



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-input', help = "Input File", dest = "inputFile", required = True)
	parser.add_argument('-output', help = "Destination Path", dest = "output", default = "")
	parser.add_argument('-k', help = "Value of K in Kmer", dest = "k", default = 8, type = int)
	parser.add_argument('-tfidf', help = "TF_IDF Cutoff Value", dest = "tf_idf_cutoff", default = 1.0, type = float)
	parser.add_argument('--useHdf', help = "Use HDF instead of sparse matrix?: True or False", dest = "useHdf", action = 'store_true', default = False)

	args = parser.parse_args()

	buildVocab(args.inputFile, args.output, args.k, args.tf_idf_cutoff, args.useHdf)



