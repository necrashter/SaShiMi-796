import torch
import glob
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sashimi import *
from wav_dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("Using device:", device)

train_dataset = SC09("./datasets/sc09/train", device=device)
# train_dataloader = [train_dataset[0]]
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

torch.manual_seed(42)

model_prefix = "./models/sc09/epoch"

model = SaShiMi(
    input_dim=1,
    hidden_dim=64,
    output_dim=256,
    state_dim=64,
    sequence_length=16000,
    block_count=8,
    encoder=Embedding(256, 64),
).to(device)

for module in model.modules():
    # isinstance doesn't work due to automatic reloading
    if type(module).__name__ == S4Base.__name__:
        module.B.requires_grad = False
        module.P.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
epoch = 0

# Load the latest model if present
models = sorted(glob.glob(model_prefix + "*.pt"))

if models:
    model_path = models[-1]
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("Using pretrained model:", model_path)
    epoch = int(model_path[len(model_prefix):-3])
    print("Epoch:", epoch)
else:
    print("Starting training from scratch")


# Update LR
for group in optimizer.param_groups:
    group["lr"] = 0.001


model.train()
while True:
    train_loss = 0.0
    for x, y in tqdm(train_dataloader, leave=False):
        y_hat = model(x)
        loss = criterion(y_hat.view(-1, 256), y.view(-1))

        loss_val = loss.detach().cpu().item()
        train_loss += loss_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    epoch += 1
    print("Epoch", epoch, "train loss:", train_loss)

    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
    }, model_prefix + f"{epoch:04d}.pt")
