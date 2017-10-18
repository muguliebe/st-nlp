import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable
import numpy as np


def to_2D(tensor, dim):
    return tensor.contiguous().view(-1, dim)


def to_3D(tensor, batch, dim):
    return tensor.contiguous().view(batch, -1, dim)


def expand(tensor, target):
    return tensor.expand_as(target)


def padded_attn(tensor, mask):
    tensor = torch.mul(tensor, mask.float())
    tensor = torch.div(tensor, expand(tensor.sum(dim=1).unsqueeze(-1), tensor))
    return tensor.unsqueeze(-1)


class InitPrev(nn.Module):
    def __init__(self, hidden_dim):
        super(InitPrev, self).__init__()
        self.hidden_dim = hidden_dim

        self.Wu = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.Wv = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)

        self.linear = nn.Linear(self.hidden_dim * 2, 1)

    def forward(self, qry_h, qm):
        '''
        qry_h: B x Q x 2*H
        '''

        batch_size = qry_h.size(0)
        qry_len = qry_h.size(1)

        qry_enc = to_2D(qry_h, self.hidden_dim * 2)
        qry_enc = to_3D(self.Wu(qry_enc), batch_size, self.hidden_dim * 2)  # B x Q x 2*H

        Vr = Variable(torch.ones(qry_len, self.hidden_dim * 2)).cuda()

        Vr_enc = self.Wv(to_2D(Vr, self.hidden_dim * 2)).unsqueeze(0)
        Vr_enc = expand(Vr_enc, qry_enc)  # B x Q x 2*H

        s = F.tanh(qry_enc + Vr_enc)  # B x Q x 2*H

        s = self.linear(to_2D(s, self.hidden_dim * 2))  # B x Q x 1
        alpha = F.softmax(s.view(batch_size, -1))
        alpha = padded_attn(alpha, qm)
        alpha = expand(alpha, qry_h)  # B x Q x 2*H

        prev_h = torch.mul(to_2D(alpha, self.hidden_dim * 2), to_2D(qry_h, self.hidden_dim * 2))
        prev_h = to_3D(prev_h, batch_size, qry_len).sum(dim=2).squeeze(-1)
        return prev_h


class AttnLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttnLayer, self).__init__()
        self.hidden_dim = hidden_dim

        self.Wq = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.Wp = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.Wr = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.Wb = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.w = nn.Linear(self.hidden_dim, 1)
        self.b = nn.Parameter(torch.FloatTensor(1, 1))

        def forward(self, doc_i, qry_h, prev_hr, qm):
            '''
            qry_h: B x Q x 2*H
            doc_i: B X 2*H
            prev_hr: B X 2*H
            qm: B x Q
            '''
            batch_size = qry_h.size(0)

        ones = Variable(torch.ones(batch_size, self.hidden_dim * 2)).cuda()  # B x 2*H
        one = Variable(torch.ones(batch_size, 1)).cuda()

        qry_h = to_2D(qry_h, self.hidden_dim * 2)  # (B x Q) x 2*H
        doc_i = to_2D(doc_i, self.hidden_dim * 2)  # (B x 1) x 2*H
        prev_hr = to_2D(prev_hr, self.hidden_dim * 2)  # (B x 1) x 2*H

        qry_enc = to_3D(self.Wq(qry_h), batch_size, self.hidden_dim)  # B x Q x H

        bias = self.Wb(ones)  # B x H
        doc_enc = self.Wp(doc_i)  # B x H
        prev_hr_enc = self.Wr(prev_hr)  # B x H

        ctx_enc = (doc_enc + prev_hr_enc + bias).unsqueeze(1)  # B x 1 x H

        G_i = F.tanh(qry_enc + expand(ctx_enc, qry_enc))  # B x Q x H
        G_i = self.w(to_2D(G_i, self.hidden_dim))  # (B x Q) x 1
        G_i = G_i.view(batch_size, -1)  # B x Q

        bias_scalar = expand(self.b.repeat(batch_size, 1), G_i)  # B x Q

        alpha = F.softmax(G_i + bias_scalar)
        alpha = padded_attn(alpha, qm)  # B x Q x 1

        return alpha


class MatchCell(nn.Module):
    def __init__(self, hidden_dim):
        super(MatchCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.grucell = nn.GRUCell(self.hidden_dim * 4, self.hidden_dim * 2)

    def forward(self, doc_i, qry_h, prev_hr, alpha):
        '''
        qry_h: B x Q x 2*H
        doc_i: B X 2*H
        prev_hr: B X 2*H
        alpha: B x Q X 1
        '''
        attn_qry = torch.bmm(qry_h.transpose(1, 2), alpha).squeeze(-1)  # B x H
        z = torch.cat((doc_i, attn_qry), 1)  # B x 4*H

        prev_hr = self.grucell(z, prev_hr)  # B x 2*H

        return prev_hr


class AnsAttnLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AnsAttnLayer, self).__init__()
        self.hidden_dim = hidden_dim

        self.V = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.Wa = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.Wb = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, hr, prev_ha, dm):
        '''
        hr: B x D x (2*H)
        prev_ha: B X (2*H)
        dm: B x D
        '''
        batch_size = hr.size(0)
        ones = Variable(torch.ones(batch_size, self.hidden_dim * 2)).cuda()  # B x 2*H

        hr = to_2D(hr, self.hidden_dim * 2)  # (B*D) x 2*H
        hr_enc = to_3D(self.V(hr), batch_size, self.hidden_dim)  # B x D x H

        prev_ha = to_2D(prev_ha, self.hidden_dim * 2)  # B X 2*H
        prev_ha_enc = self.Wa(prev_ha)  # B X H
        bias = self.Wb(ones)  # B X H

        ctx_enc = (prev_ha_enc + bias).unsqueeze(1)  # B X 1 X H
        Fi = F.tanh(hr_enc + expand(ctx_enc, hr_enc))  # B x D x H

        return Fi


class PtrNet(nn.Module):
    def __init__(self, hidden_dim):
        super(PtrNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.grucell = nn.GRUCell(self.hidden_dim * 2, self.hidden_dim * 2)

    def forward(self, hr, prev_ha, beta):
        '''
        hr: B x D x (2*H)
        prev_ha: B X (2*H)
        beta: B x D x 1
        '''
        attn_doc = torch.bmm(hr.transpose(1, 2), beta).squeeze(-1)  # B x 2H
        prev_ha = self.grucell(attn_doc, prev_ha)  # B x 2*H

        return prev_ha


class OutputLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(OutputLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, Fi):
        '''
        Fi: B x D x H
        '''
        batch_size = Fi.size(0)
        Fi = self.w(to_2D(Fi, self.hidden_dim))
        Fi = Fi.view(batch_size, -1)  # B x D

        bias = Variable(torch.ones(batch_size, 1)).cuda()  # B x 1
        bias = self.linear(bias)
        bias = expand(bias, Fi)

        beta = Fi + bias

        return beta  # B x D

