import numpy  as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable
from torch import cuda

from modules import *


class Embed(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embed, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, self.embed_dim)

    def forward(self, doc, qry):
        doc = self.embed(doc)
        qry = self.embed(qry)

        return doc, qry


class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, bidirectional):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.gru = nn.GRU(self.embed_dim, self.hidden_dim, batch_first=True, num_layers=num_layers,
                          bidirectional=bidirectional)

    def forward(self, doc, qry, doc_h0, qry_h0):
        batch_size = doc.size(0)

        qry_h, _ = self.gru(qry, qry_h0)
        doc_h, _ = self.gru(doc, doc_h0)

        return doc_h, qry_h

    def init_hidden(self, batch_size, bidirectional=False):
        hidden = next(self.parameters()).data
        if bidirectional == False:
            return Variable(hidden.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        else:
            return Variable(hidden.new(self.num_layers * 2, batch_size, self.hidden_dim).zero_())


class MatchLSTM(nn.Module):
    def __init__(self, hidden_dim, cuda=True):
        super(MatchLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.cuda = cuda

        self.initprev = InitPrev(self.hidden_dim)
        self.attn = AttnLayer(self.hidden_dim)
        self.lstm_stack = nn.ModuleList([MatchCell(self.hidden_dim) for _ in range(2)])
        self.linear = nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2)

    def forward(self, doc_h, qry_h, dm, qm):
        """
        doc_h: B x D x 2*H  //  qry_h: B x Q x 2*H
        dm: B x D  //  qm: B x Q
        """
        batch_size = doc_h.size(0)
        doc_len = doc_h.size(1)

        if self.cuda:
            hr_r = Variable(torch.zeros(batch_size, doc_len, self.hidden_dim * 2)).cuda()
            hr_l = Variable(torch.zeros(batch_size, doc_len, self.hidden_dim * 2)).cuda()
        else:
            hr_r = Variable(torch.zeros(batch_size, doc_len, self.hidden_dim * 2))
            hr_l = Variable(torch.zeros(batch_size, doc_len, self.hidden_dim * 2))

        for i in range(doc_len):
            if i == 0:
                prev_hr = self.initprev(qry_h, qm)  # B x 2*H

            alpha = self.attn(doc_h[:, i, :], qry_h, prev_hr, qm)  # B x Q x 1
            prev_hr = self.lstm_stack[0](doc_h[:, i, :], qry_h, prev_hr, alpha)  # B x 2*H
            hr_r[:, i, :] = prev_hr.unsqueeze(1)  # B x 1 x 2*H

        for i in reversed(range(doc_len)):
            if i == doc_len - 1:
                prev_hr = self.initprev(qry_h, qm)

            alpha = self.attn(doc_h[:, i, :], qry_h, prev_hr, qm)
            prev_hr = self.lstm_stack[1](doc_h[:, i, :], qry_h, prev_hr, alpha)
            hr_l[:, i, :] = prev_hr.unsqueeze(1)

        hr = torch.cat((hr_r, hr_l), 2)
        hr = self.linear(to_2D(hr, self.hidden_dim * 4))
        hr = to_3D(hr, batch_size, self.hidden_dim * 2)

        return hr


class LSTMLayer(nn.Module):
    def __init__(self, hidden_dim, bidirectional):
        super(LSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = 1
        self.gru = nn.GRU(self.hidden_dim * 2, self.hidden_dim, num_layers=self.num_layers, dropout=0.2,
                          batch_first=bidirectional, bidirectional=True)

    def forward(self, hr, hr_h0):
        """
        hr: B x D x 2*H
        """
        batch_size = hr.size(0)
        hr_h, _ = self.gru(hr, hr_h0)

        return hr_h

    def init_hidden(self, batch_size, bidirectional=False):
        hidden = next(self.parameters()).data
        if bidirectional == False:
            return Variable(hidden.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        else:
            return Variable(hidden.new(self.num_layers * 2, batch_size, self.hidden_dim).zero_())


class AnsPtr(nn.Module):
    def __init__(self, hidden_dim, cuda=True):
        super(AnsPtr, self).__init__()
        self.hidden_dim = hidden_dim
        self.cuda = cuda

        self.initprev = InitPrev(self.hidden_dim)
        self.attn = AnsAttnLayer(self.hidden_dim)
        self.out = OutputLayer(self.hidden_dim)
        self.ptr = PtrNet(self.hidden_dim)

    def forward(self, hr, qry_h, dm, qm):
        '''
        hr: B x D x 2*H
        qry_h: B x Q x 2*H
        '''
        batch_size = hr.size(0)
        doc_len = hr.size(1)

        if self.cuda:
            output = Variable(torch.zeros(batch_size, doc_len, 2)).cuda()
        else:
            output = Variable(torch.zeros(batch_size, doc_len, 2))

        for i in range(2):
            if i == 0:
                prev_ha = self.initprev(qry_h, qm)  # B x 2*H

            Fi = self.attn(hr, prev_ha, dm)  # B x D x H
            beta = self.out(Fi)  # B x D

            if i == 0:
                output[:, :, i] = beta
                beta = F.softmax(beta)
                beta = padded_attn(beta, dm)
                prev_ha = self.ptr(hr, prev_ha, beta)

            elif i == 1:
                output[:, :, i] = beta

        return output


class MatchNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, cuda, num_layers, bidirectional):
        super(MatchNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embed = Embed(vocab_size, embed_dim)
        self.encoder = Encoder(embed_dim, self.hidden_dim, num_layers, bidirectional)
        self.matchlstm = MatchLSTM(self.hidden_dim, cuda)
        self.lstm = LSTMLayer(self.hidden_dim, bidirectional)
        self.ansptr = AnsPtr(self.hidden_dim, cuda)

    def forward(self, doc, qry, dm, qm):
        batch_size = doc.size(0)

        doc, qry = self.embed(doc, qry)

        doc_h0 = self.encoder.init_hidden(batch_size, self.bidirectional)
        qry_h0 = self.encoder.init_hidden(batch_size, self.bidirectional)
        hr_h0 = self.lstm.init_hidden(batch_size, self.bidirectional)

        doc_h, qry_h = self.encoder(doc, qry, doc_h0, qry_h0)
        hr = self.matchlstm(doc_h, qry_h, dm, qm)

        hr_h = self.lstm(hr, hr_h0)
        output = self.ansptr(hr_h, qry_h, dm, qm)

        output1 = F.log_softmax((output[:, :, 0]).contiguous().view(batch_size, -1))
        output2 = F.log_softmax((output[:, :, 1]).contiguous().view(batch_size, -1))

        return output1, output2
