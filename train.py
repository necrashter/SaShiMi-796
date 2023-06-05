import os
import torch
import glob
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sashimi import *
from wav_dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("Using device:", device)

DURATION = 8
train_dataset = YoutubeMixDataset("./datasets/youtube-mix/train", duration=DURATION, device=device)
validation_dataset = YoutubeMixDataset("./datasets/youtube-mix/validation", duration=DURATION, device=device)
print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(validation_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

torch.manual_seed(42)

model_dir = "./models/ym-8l/"
if not os.path.exists(model_dir):
   os.makedirs(model_dir)

model_prefix = model_dir + "epoch"

model = SaShiMi(
    input_dim=1,
    hidden_dim=64,
    output_dim=256,
    state_dim=64,
    sequence_length=16000*DURATION,
    block_count=8,
    encoder=Embedding(256, 64),
).to(device)

for module in model.modules():
    # isinstance doesn't work due to automatic reloading
    if type(module).__name__ == S4Base.__name__:
        module.B.requires_grad = False
        module.P.requires_grad = False

print("Total parameters:", sum([i.nelement() for i in model.parameters()]))
print("Parameters to be learned:", sum([i.nelement() if i.requires_grad else 0 for i in model.parameters()]))

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
    print("Continuing from epoch:", epoch)
else:
    print("Starting training from scratch")


# Update LR
for group in optimizer.param_groups:
    group["lr"] = 0.001


model.train()
while True:
    train_loss = torch.tensor(0.0, requires_grad=False)
    for x, y in tqdm(train_dataloader, leave=False):
        y_hat = model(x)
        loss = criterion(y_hat.view(-1, 256), y.view(-1))
        train_loss += loss.detach().cpu() * x.size(dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataset)

    with torch.no_grad():
        validation_loss = torch.tensor(0.0, requires_grad=False)
        for x, y in tqdm(validation_dataloader, leave=False):
            y_hat = model(x)
            loss = criterion(y_hat.view(-1, 256), y.view(-1))
            validation_loss += loss.detach().cpu() * x.size(dim=0)
        validation_loss /= len(validation_dataset)


    epoch += 1
    print(f"EPOCH {epoch:4}",
          "| Train loss:",
          f"{train_loss.item():20}",
          f"{(train_loss / torch.tensor(2.0).log()).item():20}",
          "| Validation loss:",
          f"{validation_loss.item():20}",
          f"{(validation_loss / torch.tensor(2.0).log()).item():20}",
          )

    with open(model_dir + "validation_loss.txt", "a") as f:
        print(validation_loss.item(), file=f)
    with open(model_dir + "train_loss.txt", "a") as f:
        print(train_loss.item(), file=f)

    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
    }, model_prefix + f"{epoch:04d}.pt")
