import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mps import MPS
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPS = 1e-6
class DenseMPS(nn.Module):
	def __init__(self, input_dim, output_dim, bond_dim, feature_dim=2, nCh=3,
				 kernel=2, virtual_dim=1,
				 adaptive_mode=False, periodic_bc=False, parallel_eval=False,
				 label_site=None, path=None, init_std=1e-9, use_bias=True,
				 fixed_bias=True, cutoff=1e-10, merge_threshold=2000):
		super().__init__()
		self.input_dim = input_dim
		self.virtual_dim = bond_dim
		channel = nCh

		### Squeezing of spatial dimension in first step
		self.kScale = 4
		nCh =  self.kScale**2 * nCh
		self.input_dim = self.input_dim // self.kScale

		self.nCh = nCh
		self.ker = kernel
		iDim = (self.input_dim // (self.ker))

		feature_dim = 3*nCh

		self.dense_dim = self.virtual_dim

		self.pool1 = nn.AdaptiveAvgPool2d(iDim)
		self.BN11 = nn.BatchNorm2d(self.dense_dim,affine=True)
		self.relu11 = nn.ReLU(inplace=True)

		### First level MPS blocks
		self.module1 = nn.ModuleList([ MPS(input_dim=(self.ker)**2,
			output_dim=self.virtual_dim, 
			nCh=nCh, bond_dim=bond_dim, 
			feature_dim=feature_dim, parallel_eval=parallel_eval,
			adaptive_mode=adaptive_mode, periodic_bc=periodic_bc) 
			for i in range(torch.prod(iDim))])

		self.BN1 = nn.BatchNorm1d(self.virtual_dim,affine=True)

		
		iDim = iDim // self.ker
		#feature_dim = 3*self.virtual_dim

		self.dense_dim = 2*self.dense_dim

		self.pool2 = nn.AdaptiveAvgPool2d(iDim)
		self.BN22 = nn.BatchNorm2d(self.dense_dim, affine=True)
		self.relu22 = nn.ReLU(inplace=True)

		### Second level MPS blocks
		self.module2 = nn.ModuleList([ MPS(input_dim=self.ker**2, 
			output_dim=self.dense_dim,
			nCh=self.virtual_dim, bond_dim=bond_dim,
			feature_dim=3*self.dense_dim,  parallel_eval=parallel_eval,
			adaptive_mode=adaptive_mode, periodic_bc=periodic_bc)
			for i in range(torch.prod(iDim))])

		self.BN2 = nn.BatchNorm1d(self.dense_dim,affine=True)

		iDim = iDim // self.ker

		self.dense_dim = 2 * self.dense_dim

		self.pool3 = nn.AdaptiveAvgPool2d(iDim)
		self.BN33 = nn.BatchNorm2d(self.dense_dim, affine=True)
		self.relu33 = nn.ReLU(inplace=True)

		### Third level MPS blocks
		self.module3 = nn.ModuleList([ MPS(input_dim=self.ker**2,
			output_dim=self.dense_dim,
			nCh=self.virtual_dim, bond_dim=bond_dim,  
			feature_dim=3*self.dense_dim, parallel_eval=parallel_eval,
			adaptive_mode=adaptive_mode, periodic_bc=periodic_bc) 
			for i in range(torch.prod(iDim))])

		self.BN3 = nn.BatchNorm1d(self.dense_dim,affine=True)

		self.dense_dim = 2 * self.dense_dim

		### Final MPS block
		self.mpsFinal = MPS(input_dim=len(self.module3), 
				output_dim=self.dense_dim, nCh=1,
				bond_dim=bond_dim, feature_dim=3*self.dense_dim,
				adaptive_mode=adaptive_mode, periodic_bc=periodic_bc,
				parallel_eval=parallel_eval)
		
	def forward(self,x):
		b = x.shape[0] #Batch size 512

		# # if PCam dataset
		# r = x[:, 0]
		# g = x[:, 1]
		# d = x[:, 2]
		# x = torch.cat((r, g, d), dim=2)
		# x = torch.unsqueeze(x, dim=1)

		z = x.expand((b, self.virtual_dim, x.shape[2], x.shape[3]))

		# z = x.expand((b,self.virtual_dim,x.shape[2],x.shape[3]))

		# Increase input feature channel
		iDim = self.input_dim #32x32
		if self.kScale > 1:
			x = x.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],iDim[1]) #512x1x4x4x32x32 dim-size-step
			x = x.reshape(b,1,iDim[0],iDim[1],-1) #512x32x32x16

		# Level 1 contraction		 
		iDim = self.input_dim//(self.ker) #16x16
		x = x.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],iDim[1]).reshape(b,
					self.nCh,(self.ker)**2,-1) #512x16x4x256
		y = [ self.module1[i](x[:,:,:,i]) for i in range(len(self.module1))]
		y = torch.stack(y,dim=2) #512x5x256
		y = self.BN1(y)

		z = self.pool1(z)
		z = self.BN11(z)
		z = self.relu11(z)

		# Level 2 contraction

		y = y.view(b,self.virtual_dim,iDim[0],iDim[1])

		y = torch.cat((z,y),dim = 1)
		# y = self.relu11(y)
		z = y

		iDim = (iDim//self.ker)
		y = y.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],
				iDim[1]).reshape(b,2*self.virtual_dim,self.ker**2,-1) #512x5x4x64
		x = [ self.module2[i](y[:,:,:,i]) for i in range(len(self.module2))]
		x = torch.stack(x,dim=2)#512x5x64
		x = self.BN2(x)

		z = self.pool2(z)
		z = self.BN22(z)
		z = self.relu22(z)

		# Level 3 contraction
		x = x.view(b,2*self.virtual_dim,iDim[0],iDim[1])

		x = torch.cat((z,x),dim=1)
		# x = self.relu22(x)
		z = x

		iDim = (iDim//self.ker)
		x = x.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],
				iDim[1]).reshape(b,4*self.virtual_dim,self.ker**2,-1) #512x5x4x16
		y = [ self.module3[i](x[:,:,:,i]) for i in range(len(self.module3))]

		y = torch.stack(y,dim=2)#512x5x16
		y = self.BN3(y)

		z = self.pool3(z)
		z = self.BN33(z)
		z = self.relu33(z)

		y = y.view(b,4*self.virtual_dim,iDim[0],iDim[1])

		y = torch.cat((z,y),dim=1)
		# y = self.relu33(y)
		y = y.view(b,-1,iDim[0]*iDim[1])


		if y.shape[1] > 1:
		# Final layer
			y = self.mpsFinal(y)#512

		return y




