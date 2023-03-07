import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler


class BostonDataset(torch.utils.data.Dataset):
  '''
  Prepare the Boston dataset for regression
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


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''

  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(8, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)


if __name__ == '__main__':

  # Set fixed random number seed
  torch.manual_seed(42)
  import pandas as pd
  data = pd.read_csv('./lab1/california_housing.csv')
	# X, y
  X = data.drop(['house_value'], axis=1).values
  y = data['house_value'].values
	# Create dataset
  dataset = BostonDataset(X, y, scale_data=True)
  trainloader = torch.utils.data.DataLoader(
      dataset, batch_size=10, shuffle=True, num_workers=1)

  # Initialize the MLP
  mlp = MLP()

  # Define the loss function and optimizer
  loss_function = nn.L1Loss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

  # Run the training loop
  for epoch in range(0, 5):  # 5 epochs at maximum

    # Print epoch
    print(f'Starting epoch {epoch+1}')

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader, 0):

      # Get and prepare inputs
      inputs, targets = data
      inputs, targets = inputs.float(), targets.float()
      targets = targets.reshape((targets.shape[0], 1))

      # Zero the gradients
      optimizer.zero_grad()

      # Perform forward pass
      outputs = mlp(inputs)

      # Compute loss
      loss = loss_function(outputs, targets)

      # Perform backward pass
      loss.backward()

      # Perform optimization
      optimizer.step()

      # Print statistics
      current_loss += loss.item()
      if i % 10 == 0:
          print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 500))
          current_loss = 0.0

  # Process is complete.
  print('Training process has finished.')
