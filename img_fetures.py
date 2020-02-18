import  torch
from    pokemon import Pokemon
from    resnet import ResNet18
from    torch.utils.data import DataLoader
import  os, glob
import numpy as np
import  random, csv

import  visdom

batchsz = 32
extract_db = Pokemon('pokemon', 224, mode='extract')
extract_loader = DataLoader(extract_db, batch_size=batchsz, num_workers=2)

def resnet_cifar(model, loader):
    extract = []
    model.eval()
    for x, y, z in loader:
        x, y = x, y
        with torch.no_grad():
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = x.numpy()
            # x = x.view(x.shape[0], -1)
            # np.vstack((extract, x))
            extract.append(x)
    return extract

def main():

    model = ResNet18()

    extract = resnet_cifar(model, extract_loader)
    # extract = np.asarray(extract)
    print('提取的特征： ', type(extract))
    # print(extract)

    if not os.path.exists(os.path.join('pokemon', 'img_fetures.csv')):
        fetures = []
        for i in range(len(extract)):
            list = extract[i]
            for j in range(np.size(list, 0)):
                print(list[j])
                fetures.append(list[j])

        with open(os.path.join('pokemon', 'img_fetures.csv'), mode='w', newline='') as f:
            writer = csv.writer(f)
            for row in fetures:  # 'pokemon\\bulbasaur\\00000000.png'


                writer.writerow(row)



if __name__ == '__main__':
    main()
