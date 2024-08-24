import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from unet import Unet
from carvana_dataset import CarvanaDataset
from tqdm import tqdm

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 2
    EPOCHS = 2
    DATA_PATH = "/home/rafay/deep_learning_learning/u-net/data"
    MODEL_SAVE_PATH = "/home/rafay/deep_learning_learning/u-net/model.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CarvanaDataset(DATA_PATH)

    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8,0.2], generator=gen)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True
                                  )
    val_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True
                                  )
    
    model = Unet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_predict = model(img)
            optimizer.zero_grad()

            loss = criterion(y_predict, mask)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / idx+1
        val_running_loss = 0

        torch.cuda.empty_cache()
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_predict = model(img)
                optimizer.zero_grad()

                loss = criterion(y_predict, mask)
                val_running_loss += loss.item()

            val_loss = val_running_loss / idx+1

        print("~"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("~"*30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
