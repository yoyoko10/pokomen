import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader

from data_aggregation import Test_pictures
from mlp_net import MLP_net
from resnet import ResNet18


batchsz = 32
device = torch.device('cuda')
torch.manual_seed(1234)

class1_db = Test_pictures('pokemon', 224, mode='class_1')
class2_db = Test_pictures('pokemon', 224, mode='class_2')
class3_db = Test_pictures('pokemon', 224, mode='class_3')
class4_db = Test_pictures('pokemon', 224, mode='class_4')
class5_db = Test_pictures('pokemon', 224, mode='class_5')

class1_loader = DataLoader(class1_db, batch_size=batchsz, num_workers=4)
class2_loader = DataLoader(class2_db, batch_size=batchsz, num_workers=4)
class3_loader = DataLoader(class3_db, batch_size=batchsz, num_workers=4)
class4_loader = DataLoader(class4_db, batch_size=batchsz, num_workers=4)
class5_loader = DataLoader(class5_db, batch_size=batchsz, num_workers=4)

viz = visdom.Visdom()


def denormalize(x_hat):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # x_hat = (x-mean)/std
    # x = x_hat*std = mean
    # x: [c, h, w]
    # mean: [3] => [3, 1, 1]
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    # print(mean.shape, std.shape)
    x = x_hat * std + mean

    return x



def evalute(model, loader):
    model.eval()
    labels = []

    correct = 0
    total = len(loader.dataset)

    for x, y, z in loader:

        # viz.images(denormalize(x), nrow=6, win='batch', opts=dict(title='batch'))
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            labels.append(pred)

        # viz.text(str(labels.cpu().numpy()), win='label', opts=dict(title='batch-label'))

        correct += torch.eq(pred, y).sum().float().item()
    print(type(labels), labels)
    return correct / total



def main():

    model = ResNet18().to(device)
    model.load_state_dict(torch.load('best.mdl'))
    test_acc = evalute(model, class1_loader)
    print('test acc:', test_acc)


if __name__ == '__main__':
    main()