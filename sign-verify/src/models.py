import torch as T
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models
import config


class SiameseNetBatch(nn.Module):
    def __init__(self, inputFeatures, hiddenFeatures):
        super(SiameseNetBatch, self).__init__()
        self.internal_nodes = 512
        self.embeddings_dim = 64
        self.layers = 2
        self.initialFc = nn.Sequential(
            nn.Linear(inputFeatures, self.internal_nodes),
            nn.ReLU(),
            nn.Linear(self.internal_nodes, self.internal_nodes),
            nn.ReLU(),
            nn.Linear(self.internal_nodes, self.embeddings_dim)
        )
        self.rnnSign = nn.GRU(self.embeddings_dim, hiddenFeatures, num_layers=self.layers, batch_first=True)
        #self.rnnVsign = nn.LSTM(hiddenFeatures, hiddenFeatures, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hiddenFeatures, self.internal_nodes),
            nn.ReLU(),
            nn.Linear(self.internal_nodes, hiddenFeatures)
        )
        self.ffc = nn.Sequential(
            nn.Linear(2 * hiddenFeatures+2, self.internal_nodes),
            nn.ReLU(),
            nn.Linear(self.internal_nodes, self.internal_nodes),
            nn.ReLU(),
            nn.Linear(self.internal_nodes, 2)
        )
    
    def forward(self, sign, vsign):
        sign_embeddings = [self.initialFc(x) for x in sign]
        vsign_embeddings = [self.initialFc(x) for x in vsign]

        sign_padded = rnn_utils.pad_sequence(sign_embeddings, batch_first=True)
        vsign_padded = rnn_utils.pad_sequence(vsign_embeddings, batch_first=True)

        sign_lengths = [x.size(0) for x in sign_embeddings]
        vsign_lengths = [x.size(0) for x in vsign_embeddings]

        sign_pack = rnn_utils.pack_padded_sequence(sign_padded, sign_lengths, batch_first=True, enforce_sorted=False)
        vsign_pack = rnn_utils.pack_padded_sequence(vsign_padded, vsign_lengths, batch_first=True, enforce_sorted=False)

        sign_out, sign_hn = self.rnnSign(sign_pack)
        vsign_out, vsign_hn = self.rnnSign(vsign_pack)

        # print(sign_hn[-1].size())

        sign_final = self.fc(sign_hn[-1])
        vsign_final = self.fc(vsign_hn[-1])

        batch_size = len(sign_embeddings)
        sign_lengths_tensor = T.tensor(sign_lengths).float().view(batch_size, -1).to(config.device)
        vsign_lengths_tensor = T.tensor(vsign_lengths).float().view(batch_size, -1).to(config.device)

        print('sign_final={}, vsign_final={}'.format(sign_final.size(), vsign_final.size()))
        combined = T.cat([sign_final, vsign_final, 
            sign_lengths_tensor, vsign_lengths_tensor], dim=-1)
        # print('Combined: {}'.format(combined.size()))

        return self.ffc(combined)


class VggConvNet(nn.Module):
    def __init__(self, pretrained=True):
        super(VggConvNet, self).__init__()
        self.start_conv = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=7, padding=3)
        )
        self.vgg = models.vgg19_bn(pretrained=pretrained)
        num_ft = 1000
        self.fc = nn.Sequential(
            nn.Linear(num_ft, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.start_conv(x)
        x = self.vgg(x)
        x = self.fc(x)
        return x

