import torch as T

IMG_SIZE = (1385, 684)
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

