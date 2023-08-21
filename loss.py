import torch
from torch import nn
import torch.nn.functional as F
from op import conv2d_gradfix
from torch import autograd
import math

class AdvGLoss(nn.Module):

	def __init__(self):
		super(AdvGLoss, self).__init__()

	def forward(self, fake_pred):
		loss = F.softplus(-fake_pred).mean()
		return loss

class AdvDLoss_real(nn.Module):

	def __init__(self):
		super(AdvDLoss_real, self).__init__()

	def forward(self, real_pred):
		real_loss = F.softplus(-real_pred)
		return real_loss.mean()

class AdvDLoss_fake(nn.Module):

	def __init__(self):
		super(AdvDLoss_fake, self).__init__()

	def forward(self, fake_pred):
		fake_loss = F.softplus(fake_pred)
		return fake_loss.mean()

class AdvDLoss(nn.Module):

	def __init__(self):
		super(AdvDLoss, self).__init__()

	def forward(self, real_pred, fake_pred):
		real_loss = F.softplus(-real_pred)
		fake_loss = F.softplus(fake_pred)
		return real_loss.mean() + fake_loss.mean()


class DR1Loss(nn.Module):
    def __init__(self):
        super(DR1Loss, self).__init__()

    def forward(self,real_pred, real_img):
        print(real_pred.requires_grad, real_img.requires_grad)
        with conv2d_gradfix.no_weight_gradients():
            grad_real, = autograd.grad(
                outputs=real_pred.sum(), inputs=real_img, create_graph=True
            )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty
    