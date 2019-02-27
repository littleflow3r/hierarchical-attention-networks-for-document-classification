import torch
import torch.nn as nn

from torchtext import data, vocab
from model import hiatnn

import json
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)

dataset = data.TabularDataset(path='data/ag_news_csv/train.csv', format='csv', fields=[('label', LABEL), ('text', TEXT)])
train_dt, valid_dt = dataset.split(split_ratio=[0.8, 0.2], random_state=random.getstate())
test_dt = data.TabularDataset(path='data/ag_news_csv/test.csv', format='csv', fields=[('label', LABEL), ('text', TEXT)])

#print (len(train_dt), len(valid_dt), len(test_dt))

vec = vocab.Vectors('../../study/glove.6B.50d.txt')
TEXT.build_vocab(train_dt, vectors=vec)
print ('vector size:', TEXT.vocab.vectors.size())

with open('config.json', 'r') as fo:
    config = json.load(fo)
print (config)

bsize = config['batch_size']
gpu = config['gpu']
device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
train_it, valid_it, test_it = data.BucketIterator.splits((train_dt, valid_dt, test_dt), batch_sizes=(bsize,bsize,bsize), device=device, sort_key=lambda x: len(x.text), repeat=False)

'''
for b in train_it:
    print (b.text, b.label)
    import sys
    sys.exit()
'''
def accuracy(pred, label):
    max_pred = pred.argmax(dim=1, keepdim=True)
    correct = max_pred.squeeze(1).eq(label)
    correct = correct.sum().unsqueeze(0)
    bs = torch.LongTensor([label.shape[0]]).to(device)
    acc = correct.item() / bs.item()
    return acc

def train(model, it, lossf, optimizer):
    model.train()
    ep_loss = 0.0
    ep_acc = 0.0
    for b in it:
        optimizer.zero_grad()
        seq, label = b.text, b.label
        pred = model(seq)
        loss = lossf(pred, label)
        acc = accuracy(pred, label)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
        ep_acc += acc
    return ep_loss/len(it), ep_acc/len(it)

vocab_size = len(TEXT.vocab)
model = hiattn(vocab_size, config, vec=TEXT.vocab.vectors)
lossf = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

if gpu:
    model.to(device)

for ep in range(config['epochs']):
    print ('training epoch...', ep)
    tr_loss, tr_acc = train(model, train_it, lossf, optimizer)
    print ('TRAIN: loss %.2f acc %.1f' % (tr_loss, tr_acc))
    
    #vl_loss, vl_acc = evaluate(model, valid_it, lossf, optimizer)
    #print ('VALID: loss %.2f acc %.1f' % (vl_loss, vl_acc))

