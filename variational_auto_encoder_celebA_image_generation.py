# from wavemix_lite import WaveMixLiteImageClassification

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

from utils import save_model, load_yaml

# Set the configuration
config = load_yaml("./config/vae_celebA_config.yml")

# Training setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(config['data']['seed'])
if device == 'cuda':
  torch.cuda.manual_seed_all(config['data']['seed'])

# Set the transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(config['data']['img_size'])])

# Set the training data
train_data = datasets.CelebA(config['data']['data_path'], split='train', transform=transform, download=config['data']['download'])
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=config['data']['batch_size'],
                                           shuffle=config['data']['shuffle'],
                                           num_workers=config['data']['num_workers'],
                                           drop_last=config['data']['drop_last'])

# Set the test data
test_data = datasets.CelebA(config['data']['data_path'], split='test', transform=transform, download=config['data']['download'])
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config['data']['batch_size'],
                                          shuffle=config['data']['shuffle'],
                                          num_workers=config['data']['num_workers'],
                                          drop_last=config['data']['drop_last'])

# Check the categories
print(len(train_data.classes))

# # Set the model
# model = WaveMixLiteImageClassification(num_class=config['model']['num_class'],
#                                        num_block=config['model']['num_block'],
#                                        dim_channel=config['model']['dim_channel'],
#                                        mul_factor=config['model']['mul_factor'],
#                                        dropout=config['model']['dropout'],
#                                        device=device).to(device)
# print(model, device)

# # Set the criterion and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(),
#                         lr=config['train']['lr'],
#                         betas=config['train']['betas'],
#                         eps=config['train']['eps'],
#                         weight_decay=config['train']['weight_decay'])

# # Training
# def train(epoch, train_loader, optimizer, criterion):
#   model.train()
#   train_loss = 0.0
#   train_num = 0
#   for i, data in enumerate(train_loader, 0):
#     # get the inputs; data is a list of [inputs, labels]
#     inputs, labels = data
      
#     # Transfer data to device
#     inputs = inputs.to(device)
#     labels = labels.to(device)

#     # Model inference
#     outputs = model(inputs)
    
#     # Training
#     optimizer.zero_grad()
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()

#     # loss
#     train_loss += loss.item()
#     train_num += 1
    
#     if i % config['others']['log_period'] == 0 and i != 0:
#       print(f'[{epoch}, {i}]\t Train loss: {train_loss / train_num:.3f}')
  
#   # Average loss
#   train_loss /= train_num
  
#   return train_loss

# # Validation
# def valid(test_loader):
#   model.eval()
#   corrects = 0
#   test_num = 0

#   for _, data in enumerate(test_loader, 0):
#     # get the inputs; data is a list of [inputs, labels]
#     inputs, labels = data
      
#     # Transfer data to device
#     inputs = inputs.to(device)
#     labels = labels.to(device)

#     # Model inference
#     outputs = model(inputs)
    
#     # Calculate the accuracy
#     preds = torch.argmax(outputs.data, 1)
#     corrects += torch.sum(preds == labels.data).item()

#     # Number of the data
#     test_num += labels.size(0)
  
#   # Test accuracy
#   test_accuracy = 100 * corrects / test_num
  
#   return test_accuracy

# # Main
# if __name__ == '__main__':
#   for epoch in range(config['train']['epochs']):  # loop over the dataset multiple times
#     # Training
#     train_loss = train(epoch, train_loader, optimizer, criterion)
    
#     # Validation
#     test_accuracy = valid(test_loader)
    
#     # Print the log
#     print(f'Epoch: {epoch}\t Train loss: {train_loss:.3f}\t Valid accuracy: {test_accuracy:.3f}')
    
#     # Save the model
#     save_model(model_name=config['save']['model_name'], epoch=epoch, model=model, optimizer=optimizer, loss=train_loss, config=config)
    