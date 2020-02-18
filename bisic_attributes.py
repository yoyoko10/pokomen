import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader

from pokemon import Pokemon
from mlp_net import MLP_net

from utils import Flatten

batchsz = 32
lr = 1e-3
epochs = 11

device = torch.device('cuda')
torch.manual_seed(1234)

train_db = Pokemon('pokemon', 224, mode='train')
val_db = Pokemon('pokemon', 224, mode='val')
test_db = Pokemon('pokemon', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)

viz = visdom.Visdom()


def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y, z in loader:
        z, y = z.to(device), y.to(device)
        with torch.no_grad():
            logits = model(z)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total


def main():
    model = MLP_net().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [0], win='loss', opts=dict(title='train_loss', xlabel='batch', ylabel='loss'))
    viz.line([0], [0], win='val_acc', opts=dict(title='val_acc', xlabel='batch',ylabel='accuracy'))
    for epoch in range(epochs):

        for step, (x, y, z) in enumerate(train_loader):
            # x: [b, 3, 224, 224], y: [b]

            z, y = z.to(device), y.to(device)
            # print(z.shape, type(z))

            model.train()
            logits = model(z)

            # print('logits is:', logits.cpu().detach().numpy())
            loss = criteon(logits, y)

            # print("loss:", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1
        print('epoch :', epoch)
        if epoch % 1 == 0:

            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                # torch.save(model.state_dict(), 'best_linear.mdl')

                viz.line([val_acc], [global_step], win='val_acc', update='append')

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best_linear.mdl'))
    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)


if __name__ == '__main__':
    main()