# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:01:41 2019

@author: Rakshit
"""
global USE_CUDA # if to use CUDA
USE_CUDA = False
global MAX_SENTC # max number of sentences in the paragraphs 
MAX_SENTC = 7
global L_S # learning rate
L_S = 5.0
global L_W # learning rate
L_W = 1.0
global  lamb # alpha value in coherencce equation
lamb = 0.6

from pickle import load
import numpy as np
import pandas as pd
from numpy import linalg as LA
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from coherence_model import im2p

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return load(f)

paragraphs = load_obj('description_dict')

pickle_filename = 'features.pkl'
f = open(pickle_filename,'rb')
features = load(f) #load the features dictionary from the pickle file
f.close()

hidden_size = 256
output_size = 2048 # vocab size
vec_size = 256
coher_hidden_size = 256
topic_hidden_size = 256 
# all hidden sizes are same
nos_imgfeat = 2048 # feature vector
cont_flag = 1
n_layers_cont = 1 # 2 from the research paper but changed to 1 based on the error
n_layers_text = 1 # 1 from the research paper
n_layers_couple = 1 # 2 from the research paper but changed to 1 based on teh error
#criterion_1 = nn.CrossEntropyLoss() # loss function 1
criterion_1 = nn.MSELoss()
criterion_2 = nn.CrossEntropyLoss() # loss function 2
####### OUTPUT_SIZE has been used as per the vocabulary size for 5000 images . Change it according to number of images from preprocessing. #########
model = im2p(hidden_size = 256, output_size = 7047, vec_size = 256, coher_hidden_size = 256, topic_hidden_size = 256, nos_imgfeat = 2048, cont_flag = 1, n_layers_cont = 1, n_layers_text = 1, n_layers_couple = 1) # arguments to be passed
# optimizer
optimizer = optim.Adam(model.parameters())

for epochs in range(10):
	i = 0
	print("epoch = ",epochs)
	for img_id,feats in features.items():
		i=i+1
		print("image id : ",img_id,end = ' ')
		print("image number :",i,end = ' ')
		optimizer.zero_grad()
		loss = 0
		# paragraph = y[index(x[int(img_id)])] # get the paragraph from the csv file using dataframe
		input_variable = paragraphs[img_id]# paragraph.split('.')
		target_variable = input_variable # target variable to compare with the output to find loss
		model_hidden_st = None # Stores the hidden state vector at every step of the Sentence RNN
		nos_sentc = len(input_variable); sent_exec = 0
		sent_cand = 0
		for st in range(MAX_SENTC): # Iterate to see how many sentences the model intends to generate
			
			temp_ip = torch.from_numpy(feats)
			temp_ip = temp_ip.float()
			mod_ip = Variable(temp_ip, requires_grad=True) # Push in the Image Feature Here

			if st == 0:
				temp_hid = np.zeros(hidden_size, dtype = np.float32) # random.uniform(0, 1, (hidden_size - star_embed ) )
				temp_hid = temp_hid.reshape(1, 1, hidden_size )
				model_hidden = Variable(torch.from_numpy(temp_hid), requires_grad=True)
			else:
				mh = model_hidden_st.cpu().data.detach().numpy()
				model_hidden =  Variable(torch.from_numpy( mh[0, 0, :hidden_size].reshape(1, 1, hidden_size) ), requires_grad=True)

			# Check if Variable should be moved to GPU
			if USE_CUDA:
				mod_ip = mod_ip.cuda()
				model_hidden = model_hidden.cuda()

			# Call the model for the first time at the beginning of a sentence
			output_contstop, model_hidden = model(mod_ip, model_hidden, 'level_1') # Indicating that the first level RNN is to be used
			model_hidden_st = model_hidden.clone()
			strtstp_topv, strtstp_topi = output_contstop.data.topk(1)
			strtstp_ni = strtstp_topi[0][0]

			if strtstp_ni == 0: # So we continue
				foo = sent_cand
				sent_cand = foo + 1
		foo = loss
		loss = foo + L_S * criterion_1(torch.tensor(float(sent_cand),requires_grad = True), torch.tensor(float(nos_sentc),requires_grad = True)) # The cross-entropy loss over the number of sentences
		
		# Count the number of valid sentences in the training sample
		# val_sent = 0;
		# for st in range(nos_sentc): # Count the number of valid sentence
		# 	if len(input_variable[st]) <= 1: # If the sentence is of unit length, skip it
		# 		continue
		# 	if os.path.isfile(base_dir + 'Train/' + obj_name + '/Feat_Vec.pickle') == False: # Check if image feature file is present
		# 		return
		# 	val_sent += 1
		
		val_sent = nos_sentc
		# Create the array of topic vectors and construct the Global Topic Vector - Topic Generation Net
		gl_mh = np.zeros((1, 1, hidden_size, val_sent))
		model_hidden_st = None
		# Stack up the vectors
		for st in range(nos_sentc): # Iterate over each sentence separately
			if len(input_variable[st]) <= 1: # If the sentence is of unit length, skip it
				continue
			
			temp_ip = torch.from_numpy(feats)
			temp_ip = temp_ip.float()
			mod_ip = Variable(temp_ip, requires_grad=True)

			if sent_exec == 0: # The first sentence
				temp_hid = np.zeros(hidden_size, dtype = np.float32) # random.uniform(0, 1, (hidden_size) )
				temp_hid = temp_hid.reshape(1, 1, hidden_size )
				model_hidden = Variable(torch.from_numpy(temp_hid), requires_grad=True) # Push in the Image Feature Here #encoder_hidden
				foo = sent_exec
				sent_exec = foo + 1
			else: # All other sentences are initialized from previous sentences
				mh = model_hidden_st.cpu().data.detach().numpy()
				model_hidden =  Variable(torch.from_numpy( mh[0, 0, :hidden_size].reshape(1, 1, hidden_size) ), requires_grad=True) # Obtain the hidden state from the previous hidden state
				foo = sent_exec
				sent_exec = foo + 1

			# Check if Variable should be moved to GPU
			if USE_CUDA:
				mod_ip = mod_ip.cuda()
				model_hidden = model_hidden.cuda()

			output_contstop, model_hidden = model(mod_ip, model_hidden, 'level_1') # level_1 indicates that we are using the Senetence RNN
			model_hidden_st = model_hidden.clone()
			gl_mh[0, 0, :, sent_exec-1] = (model(model_hidden_st, None, 'topic')[0].cpu().data.detach().numpy()).reshape(1, 1, hidden_size) # Transform the hidden state to obtain the topic vector
		
		# Compute the global topic vector as a weighted average of the individual topic vectors
		glob_vec = gl_mh[0, 0, :, 0].reshape(1, 1, hidden_size)
		for i in range(1, val_sent):
			glob_vec[:, :, :] = glob_vec[:, :, :].copy() + gl_mh[:, :, :, i].reshape(1, 1, hidden_size) * (LA.norm(gl_mh[:, :, :, i].reshape(-1)) / np.sum(LA.norm(gl_mh[:, :, :, :].reshape(-1, val_sent).T, axis=1)))


		# Process the Sentence RNN
		#Previous Hidden State Vector - The Coherence Vector
		prev_vec = ( np.zeros((1, 1, hidden_size)) ).astype(np.float32)

		for st in range(nos_sentc): # Iterate over each sentence separately
			
			if len(input_variable[st]) <= 1: # If the sentence is of unit length, skip it
				continue
			ip_var = Variable(torch.LongTensor( input_variable[st] ))#, requires_grad=True) # One sentence
			op_var = Variable(torch.LongTensor( target_variable[st] ))#, requires_grad=True)
			input_length = ip_var.size()[0]
			# print("input length : ",input_length)
			target_length = op_var.size()[0]
			# print("target length : ",target_length)
			
			loc_vec = (gl_mh[:, :, :, st]).reshape(1, 1, -1) # The original topic vector for the current sentence
			# print("local vector and shape",loc_vec,loc_vec.shape)
			comb = np.add((1 - lamb) * loc_vec[0, 0, :], (lamb) * prev_vec[0, 0, :]) # Combine the current topic vector and the coherence vector from the previous sentence
			glob_vec = glob_vec.astype(np.float32)
			print("type of comb",type(comb))
			if type(comb) is not np.ndarray:
				print("/////////////////////////////////////////>>>>>>>>>>>>>>>>>>>>")
				foo = comb.numpy()
				comb = foo
			comb = comb.astype(np.float32)
			# print(glob_vec[0,0,:])
			# print("model(glob_vec[0, 0,:], comb, 'couple' )[0] : ",model(glob_vec[0, 0,:], comb, 'couple' )[0])
			mh = (((model(torch.tensor([[glob_vec[0, 0,:]]]), torch.tensor([[comb]]), 'couple' )[0]).reshape(1, 1, -1)).detach().numpy()).astype(np.float32) # Coupling Unit
			#mh = (( comb  ).reshape(1, 1, -1)).astype(np.float32)    #later check for the purpose
			#print(mh)

			# Construct the input for the first word of a sentence in the Sentence RNN
			#model_input =  Variable(torch.from_numpy( mh[0, 0, :].reshape(1, 1, feats.shape[1]) ), requires_grad=True)
			model_input =  Variable(torch.from_numpy( mh[0, 0, :]), requires_grad=True).reshape(1,1,256)
			model_hidden = Variable(torch.from_numpy(temp_hid), requires_grad=True).reshape(1,1,256)

			#print("model_input",model_input)
			if USE_CUDA:
				model_hidden = model_hidden.cuda()
				model_input = model_input.cuda()
				ip_var = ip_var.cuda()
				op_var = op_var.cuda()
				
			# Teacher forcing: Feed the target as the next input
			for di in range(1, target_length, 1):
				#print('Word Number: ' + str(di))
				print("op_var[di:di+1] :",op_var[di:di+1])
				model_output, model_hidden = model(model_input, model_hidden, 'level_2') # level_2 indicates that we want to use the Sentence RNN
				print("model output size : ",model_output.size())
				print("op_var[di:di+1] size : ",op_var[di:di+1].size())
				foo = loss
				loss = foo + L_W * criterion_2(model_output, op_var[di:di+1]) # Use the second cross-entropy term
				print("criterion 2 completed ////////////////////////////////////////////////////",di)
				embedding = nn.Embedding(7047,256)
				model_input = embedding(op_var[di:di+1]).view(1,1,-1) # Teacher forcing
				print("model input size : ",model_input.size())
				# print("model input : ",model_input)			
			# Re-initialize the previous vector
			print("entering coherence >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
			prev_vec = model(model_hidden, None, 'coher')[0].detach()
			print("coherence_completed")
		# optimizer to be added	
			print("Loss requires grad",loss.requires_grad)
			print("model parameters",model.parameters)
			print("model hidden requires grad",model_hidden.requires_grad)
		print("loss = ",loss)
		loss.backward()
		optimizer.step()
	filepath = 'model' + str(epochs) + '.pth'
	torch.save(model,filepath)
	#	return loss.data[0] * 1.0 / target_length, loss # Pass the loss information to the calling function
