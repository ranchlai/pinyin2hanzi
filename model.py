import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_ch,output_ch,kernel_size,padding,pool=True):
        super().__init__()
        self.conv = nn.Conv2d(input_ch,output_ch,kernel_size,padding=padding)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(output_ch)
        if pool:
            self.pool = nn.MaxPool2d((1,3),stride=(1,2),padding=(0,1))
        else:
            self.pool = None
    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        if self.pool is not None:
            out = self.pool(out)
        return out
        
class Model(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim,n_layers,dropout=0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                           dropout = dropout,bidirectional=True)
        
        self.conv_dim = hid_dim
        self.fc1 = nn.Linear(hid_dim*2, hid_dim*2)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(1)
        
        factor=1
        self.conv1 = ConvBlock(1,4*factor,(7,3),padding=(3,1),pool=True)
        self.conv2 = ConvBlock(4*factor,8*factor,(7,3),padding=(3,1),pool=True)
        self.conv3 = ConvBlock(8*factor,16*factor,(7,3),padding=(3,1),pool=True)
        self.conv4 = ConvBlock(16*factor,32*factor,(3,3),padding=(1,1),pool=False)
        
        self.fc2 = nn.Linear(hid_dim*8*factor, output_dim)
        self.drop2 = nn.Dropout(dropout)
        
        #self.relu = nn.LeakyReLU()
        #s#elf.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        embedded = self.embedding(src)
        max_len = src.shape[0]
        #embedded = [src sent len, batch size, emb dim]
       
        outputs, (hidden, cell) = self.rnn(embedded)
        out  = self.fc1(outputs)
        out = torch.unsqueeze(out,1)
        out = self.relu1(out)
        out = self.bn1(out)
       
       
        out = self.conv1(out)        
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        
        
        out = out.permute(0,2,1,3)
        out = out.reshape(out.shape[0],out.shape[1],-1)
        out = self.fc2(out)
        
#         out = torch.unsqueeze(out,1)
       
        out = self.log_softmax(out)
       
        
        return out#, hidden, cell
    
