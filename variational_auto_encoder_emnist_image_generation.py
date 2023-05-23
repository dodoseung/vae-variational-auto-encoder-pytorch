# from wavemix_lite import WaveMixLiteImageClassification

from variational_auto_encoder import VariationalAutoEncoder

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

from utils import save_model, load_yaml

# Set the configuration
config = load_yaml("./config/vae_emnist_config.yml")

# Training setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(config['data']['seed'])
if device == 'cuda':
  torch.cuda.manual_seed_all(config['data']['seed'])

# Set the transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(config['data']['img_size'])])

# Set the training data
train_data = datasets.EMNIST(config['data']['data_path'],
                             download=config['data']['download'],
                             split='mnist',
                             train=True,
                             transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=config['data']['batch_size'],
                                           shuffle=config['data']['shuffle'],
                                           num_workers=config['data']['num_workers'],
                                           drop_last=config['data']['drop_last'])

# Set the test data
test_data = datasets.EMNIST(config['data']['data_path'],
                            download=config['data']['download'],
                            split='mnist',
                            train=False,
                            transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config['data']['batch_size'],
                                          shuffle=config['data']['shuffle'],
                                          num_workers=config['data']['num_workers'],
                                          drop_last=config['data']['drop_last'])

# Check the categories
print(len(train_data.classes))

# Set the model
model = VariationalAutoEncoder(input_dim=(28,28,1), latent_z_dim=2,
                                enc_conv_filters=[32,64,64,64], enc_conv_kernel=[3,3,3,3], enc_conv_strides=[1,2,2,1], enc_conv_pad=[1,1,1,1],
                                dec_convt_filters=[64,64,32,1], dec_convt_kernel=[3,3,3,3], dec_convt_strides=[1,2,2,1], dec_convt_pad=[1,1,1,1]).to(device)
print(model, device)

# Set the criterion and optimizer
optimizer = optim.AdamW(model.parameters(),
                        lr=config['train']['lr'],
                        betas=config['train']['betas'],
                        eps=config['train']['eps'],
                        weight_decay=config['train']['weight_decay'])

# Training
def train(epoch, train_loader, optimizer):
  model.train()
  train_loss = 0.0
  train_num = 0
  for i, data in enumerate(train_loader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, _ = data
      
    # Transfer data to device
    inputs = inputs.to(device)

    # Model inference
    outputs, mu, log_var = model(inputs)
    
    # Calculate reconstruction loss and KL divergence
    reconst_loss = F.binary_cross_entropy(outputs, inputs, reduction='sum')
    kl_div = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)
    
    # Training
    optimizer.zero_grad()
    loss = reconst_loss + kl_div
    loss.backward()
    optimizer.step()

    # loss
    train_loss += loss.item()
    train_num += 1
    
    if i % config['others']['log_period'] == 0 and i != 0:
      print(f'[{epoch}, {i}]\t Train loss: {train_loss / train_num:.3f}')
  
  # Average loss
  train_loss /= train_num
  
  return train_loss

# Test
def valid(test_loader):
  model.eval()
  test_loss = 0
  test_num = 0

  for _, data in enumerate(test_loader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, _ = data
      
    # Transfer data to device
    inputs = inputs.to(device)

    # Model inference
    outputs, mu, log_var = model(inputs)
    
    # Calculate reconstruction loss and KL divergence
    reconst_loss = F.binary_cross_entropy(outputs, inputs, reduction='sum')
    kl_div = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)
    
    # Get loss
    loss = reconst_loss + kl_div
    test_loss += loss.item()
    test_num += inputs.size(0)
  
  # Test accuracy
  test_accuracy = test_loss / test_num
  
  return test_accuracy

# Main
if __name__ == '__main__':
  for epoch in range(config['train']['epochs']):  # loop over the dataset multiple times
    # Training
    train_loss = train(epoch, train_loader, optimizer)
    
    # Validation
    test_accuracy = valid(test_loader)
    
    # Print the log
    print(f'Epoch: {epoch}\t Train loss: {train_loss:.3f}\t Valid accuracy: {test_accuracy:.3f}')
    
    # Save the model
    save_model(model_name=config['save']['model_name'], epoch=epoch, model=model, optimizer=optimizer, loss=train_loss, config=config)
    