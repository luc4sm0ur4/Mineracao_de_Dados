import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from opacus import PrivacyEngine

# --- ETAPA 1: Definições do VAE e Carregamento de Dados (Base da Parte 1) ---

# Definir transformações para os dados
transform = transforms.ToTensor()

# Baixar e carregar o dataset de treinamento
# NOTA: O batch size deve ser consistente para o PrivacyEngine funcionar corretamente.
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
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
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

# Função de Perda (Reconstrução + Divergência KL) [cite: 10]
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- ETAPA 2: Configuração e Treinamento com Privacidade Diferencial ---

# Instanciar o modelo e o otimizador
# É importante que o modelo seja instanciado ANTES de ser passado para o PrivacyEngine
model_dp = VAE()
optimizer_dp = optim.Adam(model_dp.parameters(), lr=1e-3)

# Configurar o PrivacyEngine do Opacus
privacy_engine = PrivacyEngine()

# Anexar o PrivacyEngine ao modelo, otimizador e data loader.
# Esta é a etapa principal para "privatizar" o treinamento.
model_dp, optimizer_dp, data_loader_dp = privacy_engine.make_private(
    module=model_dp,
    optimizer=optimizer_dp,
    data_loader=train_loader,
    noise_multiplier=1.1, # Ajusta o nível de ruído [cite: 20]
    max_grad_norm=1.0,    # Ajusta o clipping de gradientes [cite: 20]
)

print("Iniciando o treinamento com Privacidade Diferencial...")

# Loop de Treinamento com DP
def train_dp(epoch):
    model_dp.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader_dp):
        # Otimizador DP, integrado pelo PrivacyEngine [cite: 19]
        optimizer_dp.zero_grad()
        
        recon_batch, mu, logvar = model_dp(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        
        # O passo do otimizador agora inclui o clipping e a adição de ruído
        optimizer_dp.step()

    # Monitorar o consumo de privacidade (epsilon) ao final de cada época [cite: 21]
    # Delta é geralmente definido como um número pequeno, menor que 1/N, onde N é o tamanho do dataset.
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    
    print(
        f"====> Epoch: {epoch} | "
        f"Average loss: {train_loss / len(data_loader_dp.dataset):.4f} | "
        f"(ε = {epsilon:.2f}, δ = 1e-5)"
    )

# Executar o treinamento por 10 épocas
NUM_EPOCHS = 10
for epoch in range(1, NUM_EPOCHS + 1):
    train_dp(epoch)

# --- ETAPA 3: Visualização dos Resultados ---

def visualize_reconstructions(model, data_loader):
    model.eval()
    data, _ = next(iter(data_loader))
    with torch.no_grad():
        recon, _, _ = model(data)

    # Plotar as imagens originais e reconstruídas para comparação
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
    fig.suptitle("Cima: Originais | Baixo: Reconstruídas com Privacidade Diferencial")
    for i in range(10):
        # Original
        axes[0, i].imshow(data[i].view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        # Reconstruída
        axes[1, i].imshow(recon[i].view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

print("\nGerando visualização das reconstruções do modelo com DP...")
visualize_reconstructions(model_dp, test_loader)