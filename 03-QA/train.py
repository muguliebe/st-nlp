###############################################################################
# Import functions and packages
###############################################################################

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from  torch.autograd import Variable

from model import MatchNet
from BatchLoader import BatchLoader
from utils import *
import Constants


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='./data/')
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=150)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', default=True)
    parser.add_argument('--glove', default='../Data/glove/glove.840B.300d.txt')
    parser.add_argument('--cuda_set', default=True)

    args = parser.parse_args()

    ###############################################################################
    # Load data
    ###############################################################################

    print("Load data file...")
    train_data, dev_data = load_data(args.data)

    print("Preparing batch loader...")
    print("============= Train ===============")
    train_loader = BatchLoader(train_data, 'train', args.cuda_set, args.batch_size)
    print("============= Valid ===============")
    dev_loader = BatchLoader(dev_data, 'dev', args.cuda_set, args.batch_size)

    # vocabulary set
    vocab_size = len(dev_data['word2idx'])
    print("============= Vocab Size ===============")
    print(vocab_size)
    print("")

    idx2word = dev_data['idx2word']

    ###############################################################################
    # Build the model
    ###############################################################################
    cuda.set_device(0)
    if args.cuda_set == True:
        model = MatchNet(vocab_size, args.embed_dim, args.hidden_dim, args.cuda_set, args.num_layers,
                         args.bidirectional).cuda()
        criterion = nn.NLLLoss().cuda()
    else:
        model = MatchNet(vocab_size, args.embed_dim, args.hidden_dim, args.cuda_set, args.num_layers,
                         args.bidirectional)
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    print("#" * 15, "Model Info", "#" * 15)
    print("Model: ", model)
    print("Criterion: ", criterion)
    print("Optimizer: ", optimizer)
    print("")

    ###############################################################################
    # Load the pretrained word embedding
    ###############################################################################

    print("loading pretrinaed word embedding ...")
    emb_file = os.path.join(args.data, 'glove_emb.pth')

    if os.path.isfile(emb_file):
        W_emb = torch.load(emb_file)
    else:
        W_emb, embed_dim = load_pretrained_embedding(dev_data['word2idx'], args.glove)
        W_emb = torch.from_numpy(W_emb).cuda()
        torch.save(W_emb, emb_file)

    if args.cuda_set:
        W_emb = W_emb.cuda()

    model.embed.embed.state_dict()['weight'].copy_(W_emb)
    model.embed.embed.state_dict()['weight'].requires_grad = False

    ###############################################################################
    # Training
    ###############################################################################

    for epoch in range(args.epoch):
        start_time = time.time()
        train_loss = AverageMeter()
        train_acc = AverageMeter()

        for i, data in enumerate(train_loader):
            model.train()

            doc = data[0];
            qry = data[1];
            anss = data[2];
            sis = data[3];
            eis = data[4];
            ids = data[5];
            dm = data[9];
            qm = data[10];

            output1, output2 = model(doc, qry, dm, qm)

            _, pred1 = output1.data.cpu().topk(1)
            _, pred2 = output2.data.cpu().topk(1)

            loss1 = criterion(output1, sis)
            loss2 = criterion(output2, eis)
            loss = loss1 + loss2

            train_loss.update(loss.data[0], doc.size(0))

            acc_tmp = accuracy(pred1.numpy(), tensor2np(sis), pred2.numpy(), tensor2np(eis), ids)
            train_acc.update(acc_tmp, doc.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("===================        Train         ======================")
            print("doc_len: ", doc.size(1))
            random_idx = randomChoice(doc.size(0))
            show_question(random_idx, doc, qry, dm, qm, anss, idx2word)
            show_answer(random_idx, doc, qry, pred1, pred2, sis, eis, idx2word)
            print("")

    message = "Train epoch: %d  iter: %d  train_loss: %1.3f  train_acc: %1.3f  elapsed: %1.3f " % (
        epoch, i, train_loss.avg, train_acc.avg, time.time() - start_time)
    print(message)
    print("")

    ###############################################################################
    # Validation
    ###############################################################################

    print("====================      Evaluation     ======================")

    val_acc = AverageMeter()
    start_time = time.time()
    model.eval()

    cor_cnt = 0;
    incor_cnt = 0;
    pad_cnt = 0;
    val_out = 0
    val_in = 0
    val_false = 0

    for j, data in enumerate(dev_loader):
        doc = data[0];
        qry = data[1];
        anss = data[2];
        sis = data[3];
        eis = data[4];
        ids = data[5];
        dm = data[9];
        qm = data[10];

        output1, output2 = model(doc, qry, dm, qm)

        _, val_pred1 = output1.data.cpu().topk(1)
        _, val_pred2 = output2.data.cpu().topk(1)

        acc_tmp = accuracy_dev(val_pred1.numpy(), sis, val_pred2.numpy(), eis, ids)
        val_acc.update(acc_tmp, doc.size(0))

    message = "Epoch: %d train_iter: %d iter: %d  val_acc: %1.3f  elapsed: %1.3f " % (
    epoch, i, j, val_acc.avg, time.time() - start_time)
    print(message)
    print("")

    ###############################################################################
    # Show the sample Q&A
    ###############################################################################

    random_idx = randomChoice(doc.size(0))
    show_question(random_idx, doc, qry, dm, qm, anss, idx2word)
    show_answer_dev(random_idx, doc, qry, val_pred1, val_pred2, sis, eis, idx2word)

    train_loss = AverageMeter()
    train_acc = AverageMeter()
    start_time = time.time()


if __name__ == '__main__':
    main()

