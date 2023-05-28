import torch
import torch.nn as nn

class MLP_clusterer(nn.Module):
    def __init__(self):
        super(MLP_clusterer, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(3072, 1500),
            nn.ReLU(),
            nn.Linear(1500, 750),
            nn.ReLU(),
            nn.Linear(750, 150)
        )

    def forward(self, x):
        x = self.fusion(x)
        return x
    
from tqdm import tqdm
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
import random
import os
import pickle

class EnricoEmbeddingDataset(Dataset):    
    def __init__(self, data_dir, file_name, random_seed=42):
        super(EnricoEmbeddingDataset, self).__init__()

        # load data
        pkl_file = os.path.join(data_dir, file_name)
        with open(pkl_file, 'rb') as f:
            corpus = pickle.load(f)
        # stable randon seed, so that the split is the same for all runs
        random.seed(random_seed)
        # shuffle
        random.shuffle(corpus)
        
        # use several lists to store the data
        self.keys = []
        self.choice = []        # the random sample choice
        self.anchor_embeddings = []
        self.positive_embeddings = []
        self.negative_embeddings = []
        for element in corpus:
            self.keys.append(element['class'])
            self.choice.append(element['choice'])
            self.anchor_embeddings.append(element['anchor_embedding'])
            self.positive_embeddings.append(element['positive_embedding'])
            self.negative_embeddings.append(element['negative_embedding'])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return [self.anchor_embeddings[idx], self.positive_embeddings[idx], self.negative_embeddings[idx]]


def get_embedding_dataloader(data_dir, file_name, batch_size, num_workers=0):
    dataset = EnricoEmbeddingDataset(data_dir, file_name)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
    return dataloader


def train(device, train_loader, net, optimizer, criterion):
    train_loss = 0.0
    for i, data in tqdm(enumerate(train_loader, 0), desc="iters"):
        anchor, positive, negative = data[0], data[1], data[2]
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device) 
        anchor_to_embedding = net(anchor)
        positive_to_embedding = net(positive)
        negative_to_embedding = net(negative) 
        optimizer.zero_grad()
        loss = criterion(anchor_to_embedding, positive_to_embedding, negative_to_embedding)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    loss = train_loss / len(train_loader)
    return loss

def operations(config: dict, net):
    # load on gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.TripletMarginLoss()

    # load data
    dataloader = get_embedding_dataloader("./image_data/dl_clustering/", 'enrico_image_triplet.pkl', batch_size=config['batch_size'])
    # initialize net
    net.to(device)
    
    print("Training " + config["net"] + "...")
    # use adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=config['weight_decay'])

    start_epoch, bench_loss = 0, 10000
    
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        loss = train(device, dataloader, net, optimizer, criterion)

        print('epoch:{}, loss:{}'.format(epoch + 1, loss * 100))
        print("------------------------------------")
        if loss < bench_loss or epoch % 10 == 0 :
            bench_loss = loss

            print('Saving model...')
            state = {
                'net': net.state_dict(),
                'epoch': epoch+1,
                'loss': loss,
            }

            if not os.path.isdir(config['weights']):
                os.mkdir(config['weights'])
            torch.save(state, config['weights'] + 'epoch_{}.ckpt'.format(epoch+1))

    print('Finished Training!')

def clustermodel():
    cluster_config = {
        'net': 'clustermodel',
        'batch_size': 64,
        'num_epochs': 300,
        'learning_rate': 0.001,
        'weight_decay': 1e-08,
        'weights': './image_data/dl_clustering/weights/',
    }
    net = MLP_clusterer()
    operations(cluster_config, net)

if __name__ == '__main__':
    clustermodel()

