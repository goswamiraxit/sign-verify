import torch as T
import models
from sklearn.preprocessing import StandardScaler
import numpy as np
import config

model = models.SiameseNetBatch(4, 1024).to(config.device)
model.load_state_dict(T.load('model-rnn.pt'))

def readLines(fp):
    return fp.readlines()

def buildRnnFingerTensor(filename):
    scaler = StandardScaler()
    vectors = [np.array([float(x.strip()) for x in line.split()]) for line in readLines(filename)[1:]]
    x = scaler.fit_transform(np.array(vectors))
    return T.from_numpy(x).float().to(config.device)


def predict(sign, vsign):
    sign_np = buildRnnFingerTensor(sign)
    vsign_np = buildRnnFingerTensor(vsign)
    output = model([sign_np], [vsign_np])
    return output.max(dim=-1)[1].cpu().item()


