######### for pytorch ##################
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

######### for internal rep ###########
from vocabulary import VocabWord
from kmer2vec import kmer2vec

###### for system and others ##########
import numpy as np
import pickle
import os
import gc
from operator import itemgetter
import tables as tb
import time
import argparse

################## for multiprocessing ########
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue

class trainModel:
	def __init__(self, vocabFilePrefix, vocabHashFile, vocabCountIndexFile, vocabularyPathCodesFile, input_path, \
		embedding_dim = 50, min_context_window = 25, max_context_window = 50, \
		max_iteration = 1500, kmer_per_sequence = 1000, batch_size = 500, device = "cpu"):

		######### the followings can be assigned directly from the parameters
		self.embedding_dim = embedding_dim
		self.min_context_window = min_context_window
		self.max_context_window = max_context_window
		self.max_iteration = max_iteration
		self.batch_size = batch_size
		self.kmer_per_sequence = kmer_per_sequence
		self.k = -1
		self.vocabSize = -1
		self.device = 'cpu'
		device = device.strip().lower()
		if 'cuda' in device:
			try:
				gpu_num = (int)(device.split(':')[1])
				if gpu_num <= torch.cuda.device_count() - 1:
					self.device = torch.device(device if torch.cuda.is_available() else "cpu")
			except:
				pass

		###### the following must be ready in pickled file ######
		self.vocabCountIndex_hdf = None
		self.vocabularyPathCodes_hdf = None
		###### load from vocabHashFile ######
		self.vocabHash = None #### should be a python dictionary
		###### load from vocabCountIndexFile
		self.colToVocab = None 
		self.vocabCounts = None
		###### load from vocabularyPathCodesFile
		self.indptr = None 
		self.indices = None
		self.data = None

		###### the following parameters are for internal use ####
		self.inputs = []
		self.inq_max_holdup = 100
		self.outq_max_holdup = 1000

		###### start loading from file #######
		self.loadVocabulary(vocabFilePrefix, vocabHashFile, vocabCountIndexFile, vocabularyPathCodesFile)
		self.loadInputFiles(input_path)

	############ the followings are for loading various components of the data #####
	def loadVocabulary(self, vocabFilePrefix, vocabHashFile, vocabCountIndexFile, vocabularyPathCodesFile):
		####### load vocabulary hashmaps and find kmer values #######
		with open(os.path.join(vocabFilePrefix, vocabHashFile), 'rb') as vocabHashFileHandle:
			self.vocabHash = pickle.load(vocabHashFileHandle)
		self.vocabSize = len(self.vocabHash)
		for km in self.vocabHash.keys():
			self.k = len(km)
			break
		print("\nValue of k = " + str(self.k) + " and total vocabulary = " + str(self.vocabSize))
		print("\nActive device: " + str(self.device))
		
		####### load vocabulary count and index arrays ########
		vocabCountIndexFile_path = os.path.join(vocabFilePrefix, vocabCountIndexFile)
		self.vocabCountIndex_hdf = tb.open_file(vocabCountIndexFile_path, "r")
		self.colToVocab = self.vocabCountIndex_hdf.root.index
		self.vocabCounts = self.vocabCountIndex_hdf.root.count
		
		###### load the sparse matrix containing the vocabular tree path and binary values ####
		vocabularyPathCodes_path = os.path.join(vocabFilePrefix, vocabularyPathCodesFile)
		self.vocabularyPathCodes_hdf = tb.open_file(vocabularyPathCodes_path, "r")
		self.indptr = self.vocabularyPathCodes_hdf.root.indptr
		self.indices = self.vocabularyPathCodes_hdf.root.indices
		self.data =  self.vocabularyPathCodes_hdf.root.data


	def loadInputFiles(self, input_path):
		for (dirpath, dirnames, filenames) in os.walk(os.path.join(input_path)):
			self.inputs.extend(filenames)
		for i in range(len(self.inputs)):
			self.inputs[i] = os.path.join(input_path, self.inputs[i])
		if len(self.inputs) == 0:
			print("No input file is provided. Now Exiting.")
			exit()

	def close_files(self):
		self.vocabCountIndex_hdf.close()
		self.vocabularyPathCodes_hdf.close()

	def getKmer(self):
		return self.k

	def getVocabSize(self):
		return self.vocabSize

	def getEmbeddingDim(self):
		return self.embedding_dim

	def setMaxIteration(self, new_max_iter):
		self.max_iteration = new_max_iter

	def reader(self, inq):
		allowed = ['a', 'c', 'g', 't']
		for inp_file in self.inputs:
			f = open(inp_file, 'r')
			sequence = ''
			for line in f:
				line = line.strip()
				if line != '' and line != '\n' and line != ' ':
					if line.startswith('>'):
						if sequence != '':
							while inq.qsize() > self.inq_max_holdup:
								pass
							inq.put(sequence)
							sequence = ''
					else:
						line = line.lower()
						trimmed_line = ""
						for ch in line:
							if ch in allowed:
								trimmed_line = trimmed_line + ch
						del line
						sequence = sequence + trimmed_line
			if sequence != '':
				inq.put(sequence)  
			f.close()
		inq.put(None)
	
	
	def prepare_batch(self, inq, outq):
		batch = []
		batch_labels = []
		###### continually take sequences from inq, process them and submit to outq ####
		while True:
			item = inq.get()
			if item is None:
				break
			sequence = item
			
			selected = []
			if len(sequence) < self.kmer_per_sequence + 200:
				for l in range(len(sequence) - self.k + 1):
					selected.append(l)
			else:
				################# divide a sequence into equal parts and randomly select kmer from each part #########
				buckets = np.linspace(start = 0, stop = len(sequence), num = self.kmer_per_sequence, dtype = np.int)
				for l in range(len(buckets) - 1):
					selected.append(np.random.randint(low = buckets[l], high = buckets[l+1] - self.k + 1))
				selected.sort()
			
			###### process kmers ####
			for i in selected:
				kmer = sequence[i : i + self.k]
				target = None
				try:
					target = self.vocabHash[kmer]
				except:
					continue
				
				###### prepare target ####
				t = list(self.indices[self.indptr[target] : self.indptr[target + 1]])
				target_path = [t for it in range(self.embedding_dim)]
				target_codes = list(self.data[self.indptr[target] : self.indptr[target + 1]])
				
				##### prepare context ######
				context = []
				cur_context_win = self.min_context_window
				
				##### if minimum and maximum context do not match then we randomly choose a context window within [min-max] ####
				if self.min_context_window != self.max_context_window:
					cur_context_win = np.random.randint(low = self.min_context_window, high = self.max_context_window + 1)
				context_start = max(i - cur_context_win, 0)
				context_end = min(i + self.k + cur_context_win, len(sequence))
				
				####### we add context of the target kmer except the target kmer #######
				for j in range(context_start, context_end - self.k):
					ckmer = sequence[j : j + self.k]
					ckmer_index = None
					try:
						ckmer_index = self.vocabHash[ckmer]
					except:
						continue
					if ckmer_index != target:
						context.append(ckmer_index)
				context.sort()
				context = [context for it in range(self.embedding_dim)]
				
				############ Now add it to the batch #####
				batch.append([context, target_path])
				batch_labels.append(target_codes)
				if len(batch) >= self.batch_size:
					while outq.qsize() >= self.outq_max_holdup:
						pass
					outq.put([batch, batch_labels])
					del batch
					del batch_labels
					batch = []
					batch_labels = []
		
		###### final check before finishing #####
		if len(batch) > 0:
			outq.put([batch, batch_labels])
			del batch
			del batch_labels
			batch = []
			batch_labels = []
		outq.put(None)


	def saveModel(self, epoch, model_dict, optimizer_dict, savedModelPath):
		state = {
			'epoch': epoch,
			'state_dict': model_dict,
			'optimizer': optimizer_dict,
		}
		torch.save(state, savedModelPath)

	def make_adaptive_lr(self, start_lr = 0.2, end_lr = 0.001):
		learning_rates = {}
		for i in range(self.max_iteration):
			learning_rates[i] = end_lr
		two_third = (int)((self.max_iteration * 2) / 3)
		b = list(np.linspace(start = start_lr, stop = end_lr, num = two_third))
		b.sort(reverse = True)
		for i in range(two_third):
			learning_rates[i] = b[i]
		return learning_rates
	
	def train(self, model, optimizer, lr, savedModelPath = None):
		#learning_rates = self.make_adaptive_lr(lr, 0.05)
		total_time = []
		for epoch in range(self.max_iteration):
			start = time.time()
			# for param_group in optimizer.param_groups:
			# 	param_group['lr'] = learning_rates[epoch]
			# print("\nStarting epoch # " + str(epoch + 1) + " with lr = " + "{:.4f}".format(learning_rates[epoch]))
			print("\nStarting epoch # " + str(epoch + 1))
			total_loss = torch.zeros(1).to(self.device)
			batch_loss = torch.zeros(1).to(self.device)
			
			###### prepare parallel queues #####
			inq = multiprocessing.Queue()
			outq = multiprocessing.Queue()
			
			####### prepare batches in parallel #####
			batchPrepareProcess = multiprocessing.Process(target = self.prepare_batch, args = (inq, outq,))
			batchPrepareProcess.daemon = True
			batchPrepareProcess.start()
			
			##### read sequences in parallel ########
			readerProcess = multiprocessing.Process(target = self.reader, args = (inq,))
			readerProcess.daemon = True
			readerProcess.start()
			
			######### collect prepared batches and do forward and backward pass ########
			while True:
				item = outq.get()
				if item is None:
					break
				
				###### get the batch #######
				batch, batch_labels = item      
				
				##### forward pass ######
				for it in range(len(batch)):
					input_item = batch[it]
					len_context = len(input_item[0][0])
					len_target =  len(input_item[1][0])
					###### prepare input contents #####
					context_gather_content = torch.LongTensor(input_item[0]).view(self.embedding_dim, len_context).to(self.device)
					target_tree_gather_content = torch.LongTensor(input_item[1]).view(self.embedding_dim, len_target).to(self.device)
					######### true label #####
					true_label = torch.FloatTensor(batch_labels[it]).to(self.device)
					#### forward pass ###
					predicted = model.forward(context_gather_content, target_tree_gather_content)
					####### loss #####
					loss = F.smooth_l1_loss(predicted, true_label, reduction = "mean")
					total_loss += loss
					batch_loss += loss

				######## back propagate #########
				if batch_loss.item() <= 0:
				  continue
				batch_loss.backward()
				optimizer.step()
				
				#### reset the gradients before doing anything ######
				model.zero_grad()
				optimizer.zero_grad()
				#print("\tBatch loss: " + str(batch_loss.item()))
				batch_loss = torch.zeros(1).to(self.device)

			###### close the parallel processes #########
			readerProcess.join()
			batchPrepareProcess.join()
			if savedModelPath is not None:
				self.saveModel(epoch + 1, model.state_dict(), optimizer.state_dict(), savedModelPath)
			duration = time.time() - start
			total_time.append(duration)
			print("\tFinished epoch # " + str(epoch + 1) + " epoch loss: " + "{:.4f}".format(total_loss.item()) + " duration: " + "{:.4f}".format(duration) + " seconds (" + "{:.4f}".format(duration / 60.00) + " minutes).")
		
		total = sum(total_time)
		avg = sum(total_time) / len(total_time)
		print("Total training time: " + str(total) + " seconds ( " + "{:.4f}".format(total / 60.00) + " minutes.)")
		print("Average training time: " + str(avg) + " seconds ( " + "{:.4f}".format(avg / 60.00) + " minutes.)")


def main(vocabFilePrefix, vocabHashFile, vocabCountIndexFile, vocabularyPathCodesFile, input_path, \
	saveModel, savedModelPath, loadModel, loadModelPath, \
	embedding_dim, min_context_window, max_context_window, learning_rate, max_iteration, \
	kmer_per_sequence, batch_size, device):

	########## create the save model directory if not exist #####
	if saveModel == True:
		if not os.path.exists(savedModelPath):
			os.makedirs(savedModelPath)

	#### create instance ####
	tm = trainModel(vocabFilePrefix, vocabHashFile, vocabCountIndexFile, vocabularyPathCodesFile, input_path, \
		embedding_dim, min_context_window, max_context_window, max_iteration, kmer_per_sequence, batch_size, device)

	state_file_name = os.path.join("states_" + str(tm.getKmer()) + "_" + str(tm.getEmbeddingDim()) + ".pytorch")

	######## create model and optimizer #####
	model = kemr2vec(tm.getVocabSize(), tm.getEmbeddingDim())
	model.to(device)
	#optimizer = optim.Adam(model.parameters(), learning_rate)
	optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9)
	
	######## load model, optimizer #######
	if loadModel == True:
		state = torch.load(os.path.join(loadModelPath, state_file_name))
		#previous_epoch = state['epoch'] 
		#max_iteration  = max_iteration - previous_epoch
		tm.setMaxIteration(max_iteration)
		model.load_state_dict(state['state_dict'])
		#optimizer.load_state_dict(state['optimizer'])
		print("Model loaded from: " + str(os.path.join(loadModelPath, state_file_name)))
		#print("Resuming from previous epoch #: " + str(previous_epoch) + ". The adjusted max # of iterations: " + str(max_iteration))

	######### now train #######
	tm.train(model, optimizer, learning_rate, os.path.join(savedModelPath, state_file_name))
	tm.close_files()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	######## parameter group 1: parameters for vocabularies #####
	parser.add_argument('-vocabFilePrefix', help = "Path of the vocabulary items.", dest = "vocabFilePrefix", required = True)
	parser.add_argument('-vocabHashFile', help = "Dictionary with vocabulary hash file.", dest = "vocabHashFile", required = True)
	parser.add_argument('-vocabCountIndexFile', help = "PyTables-HDF file with vocabulary counts and index arrays.", dest = "vocabCountIndexFile", required = True)
	parser.add_argument('-vocabularyPathCodesFile', help = "PyTables-HDF Sparse file with tree nodes and Huffman code.", dest = "vocabularyPathCodesFile", required = True)
	parser.add_argument('-input_path', help = "Directory containing all input files.", dest = "input_path", required = True)
	
	######## parameter group 2: for saving and loading learned models ####
	parser.add_argument('--saveModel', help = "Save the model states in each epoch? (True/False)", dest = "saveModel", action = 'store_true', default = True)
	parser.add_argument('-savedModelPath', help = "Path where model state is saved", dest = "savedModelPath", default = os.path.join(os.getcwd()))
	parser.add_argument('--loadModel', help = "Load the model states and continue from previous epochs? (True/False)", dest = "loadModel", action = 'store_true', default = False)
	parser.add_argument('-loadModelPath', help = "Path from where the model states will be loaded.", dest = "loadModelPath", default = os.path.join(os.getcwd()))
	
	##### parameter group 3: specific to word2vec model ######
	parser.add_argument('-embedding_dim', help = "The embedding dimension.", dest = "embedding_dim", default = 64, type = int)
	parser.add_argument('-min_context_window', help = "Minimum context window.", dest = "min_context_window", default = 30, type = int)
	parser.add_argument('-max_context_window', help = "Maximum context window.", dest = "max_context_window", default = 60, type = int)
	parser.add_argument('-learning_rate', help = "Learning rate of the model parameters.", dest = "learning_rate", default = 0.01, type = float)
	parser.add_argument('-max_iteration', help = "Maximum # of iterations.", dest = "max_iteration", default = 1500, type = int)
	
	##### parameter group 4: specific to sampling and batch size ######
	parser.add_argument('-kmer_per_sequence', help = "The # of kmers per sequence.", dest = "kmer_per_sequence", default = 1000, type = int)
	parser.add_argument('-batch_size', help = "Size of the batch.", dest = "batch_size", default = 500, type = int)
	
	##### parameter group 5: specific to system ######
	parser.add_argument('-device', help = "The device to use (cpu / cuda:0 , cuda:1 , cuda:2 , .....)", dest = "device", default = "cpu")
	
	##### parse the arguments and make variables ######
	args = parser.parse_args()
	
	##### call main method with all the parameters ####
	main(args.vocabFilePrefix, args.vocabHashFile, args.vocabCountIndexFile, args.vocabularyPathCodesFile, args.input_path, \
		args.saveModel, args.savedModelPath, args.loadModel, args.loadModelPath, \
		args.embedding_dim, args.min_context_window, args.max_context_window, args.learning_rate, args.max_iteration, \
		args.kmer_per_sequence, args.batch_size, args.device)

	



		













