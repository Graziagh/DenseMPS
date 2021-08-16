#!/usr/bin/env python3
import time
import torch
from models.densemps import DenseMPS
from models.mps import MPS
from torchvision import transforms, datasets
import pdb
from utils.dataset import load_data
from utils.tools import *
from torch.utils.data import DataLoader
import argparse

# from carbontracker.tracker import CarbonTracker

# Globally load device identifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(loader):
    ### Evaluation funcntion for validation/testing

    with torch.no_grad():
        vl_acc = 0.
        vl_loss = 0.
        labelsNp = np.zeros(1)
        predsNp = np.zeros(1)
        model.eval()
        acc_num = 0.

        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labelsNp = np.concatenate((labelsNp, labels.cpu().numpy()))

            # Inference
            scores1 = torch.sigmoid(model(inputs))

            len1 = scores1.size(0)
            len2 = scores1.size(1)

            scores2 = [sum(scores1[i]) / len2 for i in range(len1)]
            scores = torch.stack(scores2, dim=0)

            # scores = scores.squeeze(dim=1)
            preds = scores
            loss = loss_fun(scores, labels)
            predsNp = np.concatenate((predsNp, preds.cpu().numpy()))
            vl_loss += loss.item()

            ipreds = [0 if x < 0.5 else 1 for x in scores]
            ipreds = torch.tensor(ipreds)
            ipreds = ipreds.to(device)
            acc_num += float(torch.sum(ipreds.eq(labels)))

        # Compute AUC over the full (valid/test) set
        vl_auc = computeAuc(labelsNp[1:], predsNp[1:])
        vl_loss = vl_loss / len(loader)
        vl_acc = acc_num / (len(loader) * args.batch_size)

    return vl_loss, vl_acc, vl_auc


# Miscellaneous initialization
torch.manual_seed(1)
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--l2', type=float, default=0, help='L2 regularisation')
parser.add_argument('--aug', action='store_true', default=True, help='Use data augmentation')
parser.add_argument('--data_path', type=str, default='../pcam/', help='Path to data.')
parser.add_argument('--bond_dim', type=int, default=5, help='MPS Bond dimension')
parser.add_argument('--nChannel', type=int, default=3, help='Number of input channels')


args = parser.parse_args()

batch_size = args.batch_size

# LoTeNet parameters
adaptive_mode = False
periodic_bc = False

kernel = 2  # Stride along spatial dimensions
output_dim = 1  # output dimension

feature_dim = 2

logFile = time.strftime("%Y%m%d_%H_%M") + '.txt'
makeLogFile(logFile)

normTensor = 0.5 * torch.ones(args.nChannel)
# meanTensor = [0.702447, 0.546243, 0.696453]
# stdTensor = [0.238893, 0.282094, 0.216251]
### Data processing and loading....
trans_valid = transforms.Compose([transforms.Normalize(mean=normTensor, std=normTensor)])

if args.aug:
    trans_train = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=normTensor, std=normTensor)])
    print("Using Augmentation....")
else:
    trans_train = trans_valid
    print("No augmentation....")


dataset_train = load_data(split='train', data_dir=args.data_path,
                          transform=trans_train, rater=1)
dataset_valid = load_data(split='valid', data_dir=args.data_path,
                          transform=trans_valid, rater=1)
dataset_test = load_data(split='test', data_dir=args.data_path,
                         transform=trans_valid, rater=1)


num_train = len(dataset_train)
num_valid = len(dataset_valid)
num_test = len(dataset_test)
print("Num. train = %d, Num. val = %d" % (num_train, num_valid))

loader_train = DataLoader(dataset=dataset_train, drop_last=True,
                          batch_size=batch_size, shuffle=True)
loader_valid = DataLoader(dataset=dataset_valid, drop_last=True,
                          batch_size=batch_size, shuffle=False)
loader_test = DataLoader(dataset=dataset_test, drop_last=True,
                         batch_size=batch_size, shuffle=False)

# Initiliaze input dimensions
dim = torch.ShortTensor(list(dataset_train[0][0].shape[1:]))  # 图像的尺寸
nCh = int(dataset_train[0][0].shape[0])  # 图像的通道数

# load DenseMPS model
print("Using DenseMPS")
model = DenseMPS(input_dim=dim, output_dim=output_dim,
                 nCh=nCh, kernel=kernel,
                 bond_dim=args.bond_dim, feature_dim=feature_dim,
                 adaptive_mode=adaptive_mode, periodic_bc=periodic_bc, virtual_dim=1)


# Choose loss function and optimizer
loss_fun = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.l2)


nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:%d" % (nParam))
print(f"Maximum MPS bond dimension = {args.bond_dim}")
with open(logFile, "a") as f:
    print("Bond dim: %d" % (args.bond_dim), file=f)
    print("Number of parameters:%d" % (nParam), file=f)

print(f"Using Adam w/ learning rate = {args.lr:.1e}")
print("Feature_dim: %d, nCh: %d, B:%d" % (feature_dim, nCh, batch_size))

model = model.to(device)
nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

maxAuc = 0
minLoss = 1e3
convCheck = 10
convIter = 0
maxTAuc = 0

# tracker = CarbonTracker(epochs=args.num_epochs,
#                         log_dir='carbontracker/', monitor_epochs=-1)

# Let's start training!
for epoch in range(args.num_epochs):
    # tracker.epoch_start()
    running_loss = 0.
    running_acc = 0.
    t = time.time()
    model.train()
    predsNp = np.zeros(1)
    labelsNp = np.zeros(1)
    acc_num = 0.

    for i, (inputs, labels) in enumerate(loader_train):

        inputs = inputs.to(device)
        labels = labels.to(device)
        labelsNp = np.concatenate((labelsNp, labels.cpu().numpy()))

        scores1 = torch.sigmoid(model(inputs))

        len1 = scores1.size(0)
        len2 = scores1.size(1)

        scores2 = [ sum(scores1[i])/len2 for i in range(len1)]
        scores = torch.stack(scores2, dim=0)
        # scores = scores.squeeze(dim=1)
        preds = scores
        loss = loss_fun(scores, labels)

        ipreds = [0 if x < 0.5 else 1 for x in scores]
        ipreds = torch.tensor(ipreds)
        ipreds = ipreds.to(device)
        acc_num += float(torch.sum(ipreds.eq(labels)))

        with torch.no_grad():
            predsNp = np.concatenate((predsNp, preds.detach().cpu().numpy()))
            running_loss += loss

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, args.num_epochs, i + 1, nTrain, loss.item()))

    AUC = computeAuc(labelsNp, predsNp)
    ACC = acc_num / (nTrain * args.batch_size)

    # Evaluate on Validation set
    with torch.no_grad():

        vl_loss,vl_acc,vl_auc = evaluate(loader_valid)
        if vl_auc > maxAuc or vl_loss < minLoss:
            if vl_loss < minLoss:
                minLoss = vl_loss
            if vl_auc > maxAuc:
                ### Predict on test set
                ts_loss, ts_acc, ts_auc = evaluate(loader_test)
                maxAuc = vl_auc
                print('New Max: %.4f' % maxAuc)
                print('Test Set Loss:%.4f	Acc:%.4f    Auc:%.4f' % (ts_loss, ts_acc, ts_auc))
                if ts_auc > maxTAuc:
                    maxTAuc = ts_auc
                    convEpoch = epoch
                with open(logFile, "a") as f:
                    print('Test Set Loss:%.4f	Acc:%.4f    Auc:%.4f' % (ts_loss, ts_acc ,ts_auc), file=f)
            # convEpoch = epoch
            convIter = 0
        else:
            convIter += 1
        if convIter == convCheck:
            if not args.dense_net:
                print("MPS")
            else:
                print("DenseNet")
            print("Converged at epoch:%d with AUC:%.4f" % (convEpoch + 1, maxTAuc))

            break
    writeLog(logFile, epoch, running_loss / nTrain, ACC, AUC,
             vl_loss, vl_acc, vl_auc, time.time() - t)
#     tracker.epoch_end()
# tracker.stop()
