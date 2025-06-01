import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from ai.unet.DatasetClass import SegmentationDataset
import torchvision.transforms as T

from ai.unet.UNetModel import UNetModel

transform = T.Compose([
    T.ToTensor()
])

train_dataset = SegmentationDataset("../Diplom_v2-1/train/images", "./train_mask", transform=transform)
val_dataset = SegmentationDataset("../Diplom_v2-1/valid/images", "./valid_mask", transform=transform)
test_dataset = SegmentationDataset("../Diplom_v2-1/test/images", "./test_mask", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)


# обучение
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = UNetModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    command = input(": ")
    if command == "save":
        torch.save(model.state_dict(), f"runs/23_05_25/unet/unet_v2_epoch{epoch}.pth")
        print("Модель сохранена как unet_v2.pth")

    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
#

torch.save(model.state_dict(), "runs/23_05_25/unet/unet_v2.pth")
print("Модель сохранена как unet_v2.pth")
