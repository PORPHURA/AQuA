import os, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# set path
save_model_path = "./ckpts"
model_session = ''  # ENTER YOUR MODEL NAME HERE
N_REP = 10  # repetition

# ResNet parameters
res_size = 712       # ResNet image size

# training parameters
k = 2           # number of target category
epochs = 75
learning_rate = 1e-4
log_interval = 10
resume_epoch = None

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 0, 5, 1
selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)
        # print('Processed samples: %d' % N_count)

        optimizer.zero_grad()

        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(X)  # output size = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(X)

                loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(model.state_dict(), os.path.join(save_path, '3dcnn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_path, '3dcnn_optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# load UCF101 actions names
params = {'batch_size': 24, 'shuffle': True, 'num_workers': 2, 'pin_memory': True, 'drop_last': True} if use_cuda else {}  # SET BATCH_SIZE BASED ON YOUR HARDWARE

# convert labels -> category
le = LabelEncoder()
le.fit(np.arange(k))

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
enc = OneHotEncoder()
enc.fit(np.arange(k).reshape(-1, 1))


# SET YOUR OWN DATASET PATHS HERE
train_pos_path = ""
train_neg_path = ""
valid_pos_path = ""
valid_neg_path = ""

# image transformation
# transform = transforms.Compose([transforms.Resize([res_size, res_size]),
#                                 transforms.ToTensor()])

train_set, valid_set = CCUQDataset(train_pos_path, train_neg_path, selected_frames, 3), \
                       CCUQDataset(valid_pos_path, valid_neg_path, selected_frames, 3)
train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

# cache dataset
cache_path = ""  # ENTER YOUR CACHE PATH
# shutil.rmtree(cache_path)
# train_set.cache_data(cache_path)
# valid_set.cache_data(cache_path)
train_set.cache = True
train_set.cache_path = cache_path
valid_set.cache = True
valid_set.cache_path = cache_path
print('Finished caching data')


for i in range(1, N_REP):
    save_path = os.path.join(save_model_path, model_session+"_%d"%(i+1))
    os.makedirs(save_path, exist_ok=True)

    # create model
    cnn3d = ResNet_rep(in_channels=3, in_imgs=2, num_rep=len(selected_frames), num_classes=k).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn3d = nn.DataParallel(cnn3d)

    optimizer = torch.optim.Adam(cnn3d.parameters(), lr=learning_rate)   # optimize all cnn parameters

    if resume_epoch is not None:
        # save Pytorch models of best record
        cnn3d.load_state_dict(torch.load(os.path.join(save_path, '3dcnn_epoch{}.pth'.format(resume_epoch))))  # load spatial_encoder
        optimizer.load_state_dict(torch.load(os.path.join(save_path, '3dcnn_optimizer_epoch{}.pth'.format(resume_epoch))))  # save optimizer
        print("Epoch {} model loaded! Resume training!".format(resume_epoch))

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_val_losses = []
    epoch_val_scores = []

    # start training
    if resume_epoch is None:
        start_epoch = 0
    else:
        start_epoch = resume_epoch
    best_acc = 0.0
    for epoch in range(start_epoch, epochs):
        # train, test model
        train_losses, train_scores = train(log_interval, cnn3d, device, train_loader, optimizer, epoch)
        epoch_val_loss, epoch_val_score = validation(cnn3d, device, optimizer, valid_loader)

        if epoch_val_score > best_acc:
            best_acc = epoch_val_score
            print("Epoch {} best model saved!".format(epoch + 1))
            torch.save(cnn3d.state_dict(), os.path.join(save_path, '3dcnn_best.pth'))

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_val_losses.append(epoch_val_loss)
        epoch_val_scores.append(epoch_val_score)

        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_val_losses)
        D = np.array(epoch_val_scores)
        np.save('./3DCNN_epoch_training_losses.npy', A)
        np.save('./3DCNN_epoch_training_scores.npy', B)
        np.save('./3DCNN_epoch_test_loss.npy', C)
        np.save('./3DCNN_epoch_test_score.npy', D)

    # plot
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    if resume_epoch is not None:
        plt.plot(np.arange(resume_epoch+1, epochs + 1), A[:, -1])  # train loss (on epoch end)
        plt.plot(np.arange(resume_epoch+1, epochs + 1), C)         #  test loss (on epoch end)
    else:
        plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
        plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)

    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc="upper left")
    # 2nd figure
    plt.subplot(122)
    if resume_epoch is not None:
        plt.plot(np.arange(resume_epoch+1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
        plt.plot(np.arange(resume_epoch+1, epochs + 1), D)         #  test accuracy (on epoch end)
    else:
        plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
        plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
    # plt.plot(histories.losses_val)
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc="upper left")
    title = "./train_log.png"
    plt.savefig(os.path.join(save_path, title), dpi=600)
    # plt.close(fig)
    # plt.show()
