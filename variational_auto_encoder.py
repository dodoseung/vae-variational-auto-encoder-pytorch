import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

# Variational Auto Encoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim=[28,28,1], latent_z_dim=2,
                 enc_conv_filters=[32,64,64,64], enc_conv_kernel=[3,3,3,3], enc_conv_strides=[1,2,2,1], enc_conv_pad=[1,1,1,1],
                 dec_convt_filters=[64,64,32,1], dec_convt_kernel=[3,3,3,3], dec_convt_strides=[1,2,2,1], dec_convt_pad=[1,0,0,1]):
        super(VariationalAutoEncoder, self).__init__()
        
        self.encoder = Encoder(input_dim, latent_z_dim, enc_conv_filters, enc_conv_kernel, enc_conv_strides, enc_conv_pad)
        self.decoder = Decoder(self.encoder.img_size, latent_z_dim, dec_convt_filters, dec_convt_kernel, dec_convt_strides, dec_convt_pad)
        
    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_z_dim, enc_conv_filters, enc_conv_kernel, enc_conv_strides, enc_conv_pad):
        super(Encoder, self).__init__()
        # Layer information
        self.num_layers = len(enc_conv_filters)
        self.img_size = [input_dim[0], input_dim[1]]
        f = enc_conv_filters
        k = enc_conv_kernel
        s = enc_conv_strides
        p = enc_conv_pad

        # Layers
        self.convs = [nn.Conv2d(input_dim[-1], f[0], kernel_size=k[0], stride=s[0], padding=p[0])]
        self.img_size = [int(self.img_size[0] / s[0]), int(self.img_size[1] / s[0])]
        
        for i in range(1, self.num_layers):
            self.convs.append(nn.Conv2d(f[i-1], f[i], kernel_size=k[i], stride=s[i], padding= p[i]))
            self.img_size = [int(self.img_size[0] / s[i]), int(self.img_size[1] / s[i])]
        self.convs = nn.ModuleList(self.convs)

        conv_out = self.img_size[0] * self.img_size[1] * f[-1]
        self.mu = nn.Linear(conv_out, latent_z_dim)
        self.log_var = nn.Linear(conv_out, latent_z_dim)

    def forward(self, x):
        # Conv layers
        for i in range(self.num_layers):
            x = self.convs[i](x)
            x = F.leaky_relu(x)
        
        # Flatten
        x = x.flatten(start_dim=1)
        
        # Get the mu and log variance
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        # Reparameterization for back propagation
        z = self.reparameterization(mu, log_var)
        return z, mu, log_var
    
    def reparameterization(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

# Decoder
class Decoder(nn.Module):
    def __init__(self, img_size, latent_z_dim, dec_convt_filters, dec_convt_kernel, dec_convt_strides, dec_convt_pad):
        super(Decoder, self).__init__()
        self.num_layers = len(dec_convt_filters)
        f = dec_convt_filters
        k = dec_convt_kernel
        s = dec_convt_strides
        p = dec_convt_pad
        self.init_dim = f[0]

        conv_in = img_size[0] * img_size[1] * f[0]
        self.dense = nn.Linear(latent_z_dim, conv_in)
        
        self.convTs = [nn.ConvTranspose2d(f[0], f[0], kernel_size=k[0], stride=s[0], padding=p[0])]
        for i in range(1, self.num_layers):
            self.convTs.append(nn.ConvTranspose2d(f[i-1], f[i], kernel_size=k[i], stride=s[i], padding=p[i]))
        self.convTs = nn.ModuleList(self.convTs)

    def forward(self, x):
        # Dense
        x = self.dense(x)

        # Reshape
        img_size = int((int(len(x[1]) / self.init_dim))**(1/2))
        x = x.view(-1, self.init_dim, img_size, img_size)

        # Conv Transpose layers
        for i in range(self.num_layers):
            x = self.convTs[i](x)
            x = F.leaky_relu(x)
            
        # Activation
        x = F.sigmoid(x)
        return x