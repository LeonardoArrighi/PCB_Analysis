import torch
from torch import nn
import timm
import argparse
import os

import data
import optimizers
from utils import accuracy, AverageMeter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_train", type=int, default=64, help="batch size for training")
    parser.add_argument("--batch_test", type=int, default=64, help="batch size for test")
    parser.add_argument("--data_dir", type=str, default="data/", help="data directory")
    parser.add_argument("--epochs", type=int, default=600, help="number of epochs")
    parser.add_argument("--dropout_p", type=float, default=0.1, help="Dropout probability for final FC layers")
    parser.add_argument("--device", default=None, help="Device to run on")
    parser.add_argument("--checkpoint_folder", type=str, default="checkpoints/", help="Folder to save checkpoints")
    parser.add_argument("--model_save_folder", type=str, default="models/", help="Folder to save models")
    parser.add_argument("--test_folder", type=str, default="test/", help="Folder to save test set")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_folder, exist_ok=True)
    os.makedirs(args.model_save_folder, exist_ok=True)
    os.makedirs(args.test_folder, exist_ok=True)
    
    # model
    model = timm.create_model('resnet50d', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    # modify classifier
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(args.dropout_p),
        nn.Linear(512, 2),
        # nn.LogSoftmax(dim=1)
    )
    model.fc.requires_grad = True

    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optimizers.RAdam(model.fc.parameters())

    # load on the device
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load dataloaders splitting dataset
    train_loader, test_loader, val_loader = data.get_dataloaders(args.data_dir, batch_size_train=args.batch_train, batch_size_test=args.batch_test)
    
    # open to write the test set images randomly selected
    f = open(os.path.join(args.test_folder, f"test_set.txt"), "w")
    for i in test_loader.dataset.indices:
        f.write(str(test_loader.dataset.dataset.imgs[i][0])+'\n')
    f.close()

    # train
    for epoch in range(args.epochs):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(accuracy(outputs, labels).item(), inputs.size(0))
        
        loss_meter_valid = AverageMeter()
        acc_meter_valid = AverageMeter()
        with torch.no_grad():
            model.eval()
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_meter_valid.update(loss.item(), inputs.size(0))
                acc_meter_valid.update(accuracy(outputs, labels).item(), inputs.size(0))
            
        print(f"Epoch {epoch}/{args.epochs}, Train Loss: {loss_meter.avg:.4f}, Train Accuracy: {acc_meter.avg:.4f}, Validation Loss: {loss_meter_valid.avg:.4f}, Validation Accuracy: {acc_meter_valid.avg:.4f}")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_meter.avg,
                "accuracy": acc_meter.avg,
                "epoch": epoch
            },
            os.path.join(args.checkpoint_folder, f"checkpoint.pth")
        )
    
    # test
    loss_meter_test = AverageMeter()
    acc_meter_test = AverageMeter()
    with torch.no_grad():
        model.eval()
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_meter_test.update(loss.item(), inputs.size(0))
            acc_meter_test.update(accuracy(outputs, labels).item(), inputs.size(0))
    
    print(f"**TEST** Test Loss: {loss_meter_test.avg:.4f}, Test Accuracy: {acc_meter_test.avg:.4f}")

    # save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "test_loss": loss_meter_test.avg,
            "test_accuracy": acc_meter_test.avg
        },
        os.path.join(args.model_save_folder, f"model.pth")
    )
