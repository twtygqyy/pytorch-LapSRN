import argparse, os
import pdb
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from lapsrn_wgan import _netG, _netD, L1_Charbonnier_loss
from dataset import DatasetFromHdf5
from torchvision import models, transforms
import torch.utils.model_zoo as model_zoo

# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN WGAN")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=400, help="number of epochs to train for")
parser.add_argument('--lrG', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--lrD', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument("--step", type=int, default=50, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

def main():

    global opt, model 
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromHdf5("data/lap_pry_x4_small.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print('===> Building generator model')
    netG = _netG()

    print('===> Building discriminator model')    
    netD = _netD()

    print('===> Loading VGG model') 
    model_urls = {
        "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
    }

    netVGG = models.vgg19()
    netVGG.load_state_dict(model_zoo.load_url(model_urls['vgg19']))

    weight = torch.FloatTensor(64,1,3,3)
    parameters = list(netVGG.parameters())
    for i in range(64):
        weight[i,:,:,:] = parameters[0].data[i].mean(0)
    bias = parameters[1].data

    class _content_model(nn.Module):
        def __init__(self):
            super(_content_model, self).__init__()
            self.conv = conv2d = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.feature = nn.Sequential(*list(netVGG.features.children())[1:-1])
            self._initialize_weights()

        def forward(self, x):
            out = self.conv(x)
            out = self.feature(out)
            return out

        def _initialize_weights(self):
            self.conv.weight.data.copy_(weight)
            self.conv.bias.data.copy_(bias)

    netContent = _content_model()

    print('===> Building Loss')
    criterion = L1_Charbonnier_loss()

    print("===> Setting GPU")
    if cuda:
        netG = netG.cuda()
        netD = netD.cuda()
        netContent = netContent.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            netG.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            netG.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1): 
        train(training_data_loader, optimizerG, optimizerD, netG, netD, netContent, criterion, epoch)
        save_checkpoint(netG, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizerG, optimizerD, netG, netD, netContent, criterion, epoch):

    netG.train()
    netD.train()

    one = torch.FloatTensor([1.])
    mone = one * -1
    content_weight = torch.FloatTensor([1.])
    adversarial_weight = torch.FloatTensor([1.])

    for iteration, batch in enumerate(training_data_loader, 1):

        input, label_x2, label_x4 = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            label_x2 = label_x2.cuda()
            label_x4 = label_x4.cuda()
            one, mone, content_weight, adversarial_weight = one.cuda(), mone.cuda(), content_weight.cuda(), adversarial_weight.cuda()

        ############################
        # (1) Update D network: loss = D(x)) - D(G(z))
        ###########################

        # train with real
        errD_real = netD(label_x4)
        errD_real.backward(one, retain_graph=True)

        # train with fake
        input_G = Variable(input.data, volatile = True)
        fake_x4 = Variable(netG(input_G)[1].data)
        fake_D = fake_x4
        errD_fake = netD(fake_D)

        errD_fake.backward(mone)

        errD = errD_real - errD_fake
        optimizerD.step()

        for p in netD.parameters(): # reset requires_grad
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        netD.zero_grad()
        netG.zero_grad()
        netContent.zero_grad()
        
        ############################
        # (2) Update G network: loss = D(G(z))
        ###########################      
      
        fake_D_x2, fake_D_x4 = netG(input)
        content_fake_x2 = netContent(fake_D_x2)
        content_real_x2 = netContent(label_x2)
        content_real_x2 = Variable(content_real_x2.data)       
        content_loss_x2 = criterion(content_fake_x2, content_real_x2)
        content_loss_x2.backward(content_weight, retain_graph=True)

        content_fake_x4 = netContent(fake_D_x4)
        content_real_x4 = netContent(label_x4)
        content_real_x4 = Variable(content_real_x4.data)       
        content_loss_x4 = criterion(content_fake_x4, content_real_x4)
        content_loss_x4.backward(content_weight, retain_graph=True)

        content_loss = content_loss_x2 + content_loss_x4

        adversarial_loss = netD(fake_D_x4)
        adversarial_loss.backward(adversarial_weight)

        optimizerG.step()

        netD.zero_grad()
        netG.zero_grad()
        netContent.zero_grad()
        if iteration%10 == 0:
            print("===> Epoch[{}]({}/{}): LossD: {:.10f} [{:.10f} - {:.10f}] LossG: [{:.10f} + {:.10f}]".format(epoch, iteration, len(training_data_loader), 
                  errD.data[0], errD_real.data[0], errD_fake.data[0], adversarial_loss.data[0], content_loss.data[0]))   

def save_checkpoint(model, epoch):
    model_folder = "checkpoint/"
    model_out_path = model_folder + "lapwgan_model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()