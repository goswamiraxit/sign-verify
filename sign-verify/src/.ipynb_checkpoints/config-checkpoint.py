import torch as T

device = T.device('cuda' if T.cuda.is_available() else 'cpu')
IMG_SIZE = (1385, 684)
BATCH_SIZE = 16
LEARNING_RATE = 1e-5

print('Using device {}'.format(device))

