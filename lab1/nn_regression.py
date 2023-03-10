import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

model_src_file = "./lab1/result/8-64-1.pth"

class CaliforniaDataset(torch.utils.data.Dataset):
    '''
    Prepare the California dataset for regression
    '''
    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class MultiLayerPerceptron(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self, variables):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(variables, 64),
            #nn.ReLU(),
            #nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)

def train(X, y):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set fixed random number seed
    torch.manual_seed(42)

    # Create dataset
    dataset = CaliforniaDataset(X, y, scale_data=False)
    trainloader = DataLoader(
        dataset, batch_size=100, shuffle=True, num_workers=1)
    
    # Create model
    model = MultiLayerPerceptron(X.shape[1]).to(device)

    # Define loss function
    loss_fn = nn.MSELoss().to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 120
    # Train model
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for i, (X, y) in enumerate(trainloader):
            inputs, targets = X.to(device).float(), y.float().to(device)
            targets = targets.reshape((targets.shape[0], 1)).to(device)
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Iteration {i}, Loss: {loss.item():.4f}')
    
    # Save model
    torch.save(model.state_dict(), model_src_file)

def test(X, y):
    # Load model
    model = MultiLayerPerceptron(X.shape[1])
    model.load_state_dict(torch.load(model_src_file))
    model.eval()

    # Evaluate model with R2 score
    y_pred = model(torch.from_numpy(X).float())
    y_pred = y_pred.cpu().detach().numpy()
    r2 = r2_score(y, y_pred)
    # Calculate adjusted r2
    n = len(y)
    p = X.shape[1]
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(r2_adj)


if __name__ == '__main__':
    
    train_data = pd.read_csv('./lab1/dataset/train_set.csv')
    test_data = pd.read_csv('./lab1/dataset/test_set.csv')
    # X, y
    X = train_data.drop(['house_value'], axis=1).values
    y = train_data['house_value'].values
    # X_test, y_test
    X_test = test_data.drop(['house_value'], axis=1).values
    y_test = test_data['house_value'].values

    # Train model
    train(X, y)

    # Test model
    print("test r2 score")
    test(X_test, y_test)


    