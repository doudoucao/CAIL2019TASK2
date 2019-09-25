import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False

class Caps_Layer(nn.Module):

    def __init__(self, batch_size, input_dim_capsule, num_capsule, dim_capsule, routings, kernel_size=(9, 1),
                 share_weight=True, activation='default', **kwargs):
        super(Caps_Layer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weight
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule))
            )
        else:
            self.W = nn.Parameter(torch.randn(batch_size, input_dim_capsule, self.num_capsule*self.dim_capsule))


    def forward(self, x):
        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3) # batch_size, num_capsule, input_num_capsule, dim_capsule
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0]) # batch_size, num_capsule, input_num_capsule

        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))
        return outputs

    def squash(self, x, axis=-1):
        s_squared_norm = (x**2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + 1e-7)
        return x/scale


class CapsNet(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32*6*6, in_channels=8, out_channels=16):
        super(CapsNet, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x]*self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W]*batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij]*batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j]*self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)
        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor