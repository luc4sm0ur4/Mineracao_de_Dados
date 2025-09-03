import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Definir transformações para os dados
transform = transforms.ToTensor()

# Baixar e carregar o dataset de treinamento
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Baixar e carregar o dataset de teste
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Definição da arquitetura do VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, 20)  # Camada para a média (mu)
        self.fc_logvar = nn.Linear(400, 20)  # Camada para o log da variância (logvar)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()  # Sigmoid para normalizar os pixels entre 0 e 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x.view(-1, 784))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Função de Perda (Reconstrução + Divergência KL)
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Instanciar o modelo e o otimizador
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Loop de Treinamento
def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Executar o treinamento
for epoch in range(1, 11):
    train(epoch)

# Visualizar reconstruções
def visualize_reconstructions(model, data_loader):
    model.eval()
    data, _ = next(iter(data_loader))
    with torch.no_grad():
        recon, _, _ = model(data)

    # Plotar as imagens originais e reconstruídas
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
    for i in range(10):
        # Original
        axes[0, i].imshow(data[i].view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        # Reconstruída
        axes[1, i].imshow(recon[i].view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

visualize_reconstructions(vae, test_loader)