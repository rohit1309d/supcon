# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
	"""
	Build a SimCLR-based model with a base encoder, and two MLPs
   
	"""
	def __init__(self, base_encoder, dim=256, mlp_dim=2048, T=0.1, loss_type='dcl', N=50000, num_proj_layers=2, device=None):
		"""
		dim: feature dimension (default: 256)
		mlp_dim: hidden dimension in MLPs (default: 4096)
		T: softmax temperature (default: 1.0)
		"""
		super(SimCLR, self).__init__()
		self.T = T
		self.N = N
		self.loss_type = loss_type
		
		# build encoders
		self.base_encoder = base_encoder(num_classes=mlp_dim)
		
		# build non-linear projection heads
		self._build_projector_and_predictor_mlps(dim, mlp_dim)

		# sogclr 
		if not device:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = device 
		
		# for DCL
		self.u = torch.zeros(N).reshape(-1, 1) #.to(self.device) 
		self.v = torch.zeros(N).reshape(-1, 1) #.to(self.device) 
		self.LARGE_NUM = 1e9


	def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
		pass
	
	def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
		mlp = []
		for l in range(num_layers):
			dim1 = input_dim if l == 0 else mlp_dim
			dim2 = output_dim if l == num_layers - 1 else mlp_dim

			mlp.append(nn.Linear(dim1, dim2, bias=False))

			if l < num_layers - 1:
				mlp.append(nn.BatchNorm1d(dim2))
				mlp.append(nn.ReLU(inplace=True))
			elif last_bn:
				# follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
				# for simplicity, we further removed gamma in BN
				mlp.append(nn.BatchNorm1d(dim2, affine=False))

		return nn.Sequential(*mlp)

	def dynamic_contrastive_loss(self, hidden1, hidden2, labels_idx=None, index=None, gamma=0.9, distributed=True):
		# Get (normalized) hidden1 and hidden2.
		hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
		batch_size = hidden1.shape[0]

		# Gather hidden1/hidden2 across replicas and create local labels.
		if distributed:  
			hidden1_large = torch.cat(all_gather_layer.apply(hidden1), dim=0) # why concat_all_gather()
			hidden2_large =  torch.cat(all_gather_layer.apply(hidden2), dim=0)
			labels_large = torch.cat(all_gather_layer.apply(labels_idx), dim=0)
			enlarged_batch_size = hidden1_large.shape[0]

			if labels_idx is None:			   
				labels_idx = (torch.arange(batch_size, dtype=torch.long) + batch_size  * torch.distributed.get_rank()).to(self.device) 
				labels = F.one_hot(labels_idx, enlarged_batch_size*2).to(self.device) 
				masks  = F.one_hot((torch.arange(batch_size, dtype=torch.long) + batch_size  * torch.distributed.get_rank()), enlarged_batch_size).to(self.device) 
			else:
				masks  = F.one_hot((torch.arange(batch_size, dtype=torch.long) + batch_size  * torch.distributed.get_rank()), enlarged_batch_size).to(self.device) 
				labels = torch.cat([(labels_idx[:, None] == labels_large[None, :]),(labels_idx[:, None] == labels_large[None, :])], dim=1).float().to(self.device) 
			batch_size = enlarged_batch_size
		else:
			hidden1_large = hidden1
			hidden2_large = hidden2
			masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(self.device) 
			if labels_idx is None:	
				labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).to(self.device) 
			else:
				labels = torch.cat([(labels_idx.unsqueeze(0) == labels_idx.unsqueeze(1)),(labels_idx.unsqueeze(0) == labels_idx.unsqueeze(1))], dim=1).float().to(self.device) 

		logits_aa = torch.matmul(hidden1, hidden1_large.T)
		logits_aa = logits_aa - masks * self.LARGE_NUM
		logits_bb = torch.matmul(hidden2, hidden2_large.T)
		logits_bb = logits_bb - masks * self.LARGE_NUM
		logits_ab = torch.matmul(hidden1, hidden2_large.T)
		logits_ba = torch.matmul(hidden2, hidden1_large.T)

		#  SogCLR
		neg_mask = 1-labels
		logits_ab_aa = torch.cat([logits_ab, logits_aa ], 1)
		logits_ba_bb = torch.cat([logits_ba, logits_bb ], 1)

		neg_logits1 = torch.exp(logits_ab_aa /self.T)*neg_mask   #(B, 2B)
		neg_logits2 = torch.exp(logits_ba_bb /self.T)*neg_mask
		pos_logits1 = torch.exp(logits_ab_aa /self.T)*labels   #(B, 2B)
		pos_logits2 = torch.exp(logits_ba_bb /self.T)*labels

		# u init    
		if self.u[index.cpu()].sum() == 0:
			gamma = 1
			
		u1 = (1 - gamma) * self.u[index.cpu()].cuda() + gamma * torch.sum(neg_logits1, dim=1, keepdim=True)/(2*(batch_size-1))
		u2 = (1 - gamma) * self.u[index.cpu()].cuda() + gamma * torch.sum(neg_logits2, dim=1, keepdim=True)/(2*(batch_size-1))

		# this sync on all devices (since "hidden" are gathering from all devices)
		if distributed:
			u1_large = concat_all_gather(u1)
			u2_large = concat_all_gather(u2)
			index_large = concat_all_gather(index)
			self.u[index_large.cpu()] = (u1_large.detach().cpu() + u2_large.detach().cpu())/2 
		else:
			self.u[index.cpu()] = (u1.detach().cpu() + u2.detach().cpu())/2 

		p_pos_weights1 = (pos_logits1/u1).detach()
		p_pos_weights2 = (pos_logits2/u2).detach()

		v1 = (1 - gamma) * self.v[index.cpu()].cuda() + gamma * torch.sum(p_pos_weights1, dim=1, keepdim=True)/torch.sum(labels, dim=1, keepdim=True)
		v2 = (1 - gamma) * self.v[index.cpu()].cuda() + gamma * torch.sum(p_pos_weights2, dim=1, keepdim=True)/torch.sum(labels, dim=1, keepdim=True)

		# this sync on all devices (since "hidden" are gathering from all devices)
		if distributed:
			v1_large = concat_all_gather(v1)
			v2_large = concat_all_gather(v2)
			index_large = concat_all_gather(index)
			self.v[index_large.cpu()] = (v1_large.detach().cpu() + v2_large.detach().cpu())/2 
		else:
			self.v[index.cpu()] = (v1.detach().cpu() + v2.detach().cpu())/2 

		def df2(labels, logits, p_pos_weights, neg_logits, u):
			exp_logits = p_pos_weights*logits
			df1 = torch.sum(neg_logits*logits, dim=1, keepdim=True)
			normalised_logits = (p_pos_weights*df1)/u - exp_logits
			return -torch.sum(labels*normalised_logits, dim=1, keepdim=True)
		loss_a = -torch.sum(df2(labels, logits_ab_aa, p_pos_weights1, neg_logits1, u1)/v1, dim=1)
		loss_b = -torch.sum(df2(labels, logits_ba_bb, p_pos_weights2, neg_logits2, u2)/v2, dim=1)
		loss = (loss_a + loss_b).mean()
		return loss
	
	def forward(self, x1, x2, labels, index, gamma):
		"""
		Input:
			x1: first views of images
			x2: second views of images
			index: index of image
			gamma: moving average of sogclr 
		Output:
			loss
		"""
		# compute features
		h1 = self.base_encoder(x1)
		h2 = self.base_encoder(x2)
		loss = self.dynamic_contrastive_loss(h1, h2, labels, index, gamma) 
		return loss


class SimCLR_ResNet(SimCLR):
	def _build_projector_and_predictor_mlps(self, dim, mlp_dim, num_proj_layers=2):
		hidden_dim = self.base_encoder.fc.weight.shape[1]
		del self.base_encoder.fc  # remove original fc layer
			
		# projectors
		# TODO: increase number of mlp layers 
		self.base_encoder.fc = self._build_mlp(num_proj_layers, hidden_dim, mlp_dim, dim)
	   


# utils
@torch.no_grad()
def concat_all_gather(tensor):
	"""
	Performs all_gather operation on the provided tensors.
	*** Warning ***: torch.distributed.all_gather has no gradient.
	"""
	tensors_gather = [torch.ones_like(tensor)
		for _ in range(torch.distributed.get_world_size())]
	torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

	output = torch.cat(tensors_gather, dim=0)
	return output


class all_gather_layer(torch.autograd.Function):
	"""Gather tensors from all process, supporting backward propagation."""

	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		output = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
		torch.distributed.all_gather(output, input)
		return tuple(output)

	@staticmethod
	def backward(ctx, *grads):
		(input,) = ctx.saved_tensors
		grad_out = torch.zeros_like(input)
		grad_out[:] = grads[torch.distributed.get_rank()]
		return grad_out
