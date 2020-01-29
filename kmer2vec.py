######### for pytorch ##################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class kmer2vec(nn.Module):
	def __init__(self, vocabSize, embedding_dim = 100):
		super(word2vec, self).__init__()
		######### Initialize the weight vectors ###########
		self.words = nn.Parameter(torch.randn((embedding_dim, vocabSize), dtype = torch.float32), requires_grad=True) ######### the original word vectors ######
		self.tree = nn.Parameter(torch.zeros((embedding_dim, vocabSize), dtype = torch.float32), requires_grad=True) #### the tree node vectors for hierarchical softmax #####

	def forward(self, context_gather_content, target_tree_gather_content):
		######### calculate context mean #########
		context_items = torch.gather(self.words, 1, context_gather_content)
		context_mean = torch.mean(context_items, 1)
		######## prepare the target #########
		target_tree_items = torch.gather(self.tree, 1, target_tree_gather_content)
		###### do matrix multiplication ###########
		return torch.sigmoid(torch.matmul(context_mean , target_tree_items))