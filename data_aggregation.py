import  torch
import  os, glob
import  random, csv

from    torch.utils.data import Dataset, DataLoader

from    torchvision import transforms
from    PIL import Image
import  numpy as np


class Test_pictures(Dataset):

    def __init__(self, root, resize, mode):
        super(Test_pictures, self).__init__()

        self.root = root
        self.resize = resize
        self.images = []
        self.labels = []
        self.atts = []
        self.images0, self.labels0, self.images1, self.labels1, \
        self.images2, self.labels2, self.images3, self.labels3,self.images4, self.labels4,\
        self.atts0, self.atts1, self.atts2, self.atts3, self.atts4= self.load_csv('images.csv')
        self.length = 100
        if mode =='class_1':
            self.images = self.images0[int(0.8*len(self.images0)):]
            self.labels = self.labels0[int(0.8*len(self.labels0)):]
            self.atts = self.atts0[int(0.8*len(self.atts0)):]
        elif mode =='class_2':
            self.images = self.images1[int(0.8*len(self.images1)):]
            self.labels = self.labels1[int(0.8*len(self.labels1)):]
            self.atts = self.atts1[int(0.8*len(self.atts1)):]
        elif mode == 'class_3':
            self.images = self.images2[int(0.8*len(self.images2)):]
            self.labels = self.labels2[int(0.8*len(self.labels2)):]
            self.atts = self.atts2[int(0.8*len(self.atts2)):]
        elif mode == 'class_4':
            self.images = self.images3[int(0.8*len(self.images3)):]
            self.labels = self.labels3[int(0.8*len(self.labels3)):]
            self.atts = self.atts3[int(0.8*len(self.atts3)):]
        else:
            self.images = self.images4[int(0.8*len(self.images4)):]
            self.labels = self.labels4[int(0.8*len(self.labels4)):]
            self.atts = self.atts4[int(0.8*len(self.atts4)):]


    def load_csv(self,filename):
        images0, labels0, images1, labels1, images2, labels2= [] , [], [] , [],[] ,[]
        images3, labels3, images4, labels4 = [], [], [], []
        atts0, atts1, atts2, atts3, atts4 = [], [], [], [], []
        with open(os.path.join(self.root,filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img = row[0]
                label = row[1]
                label = int(label)
                att = row[2:]
                att = np.asarray(att)
                att = att.astype(np.float)


                if label == 0:
                    images0.append(img)
                    labels0.append(label)
                    atts0.append(att)
                elif label == 1:
                    images1.append(img)
                    labels1.append(label)
                    atts1.append(att)
                elif label == 2:
                    images2.append(img)
                    labels2.append(label)
                    atts2.append(att)
                elif label == 3:
                    images3.append(img)
                    labels3.append(label)
                    atts3.append(att)
                else:
                    images4.append(img)
                    labels4.append(label)
                    atts4.append(att)

        return images0, labels0, images1, labels1, images2, labels2, images3, labels3, images4, labels4, \
               atts0, atts1, atts2, atts3, atts4




    def __len__(self):

        return len(self.images)

    def denormalize(self, x_hat):

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

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0
        img, label, att = self.images[idx], self.labels[idx], self.atts[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)
        att = torch.tensor(att, dtype=torch.float32)

        return img, label, att

def main():

    import  visdom
    import  time
    import  torchvision

    viz = visdom.Visdom()
    db = Test_pictures('pokemon', 224, 'class_3')

    x, y, z = next(iter(db))
    # print('sample:', x.shape, y.shape, y)

    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=18, shuffle=True, num_workers=8)

    for x, y, z in loader:
        viz.images(db.denormalize(x), nrow=6, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-label'))
        viz.text(str(z.numpy()), win='attibutes', opts=dict(title='batch_attributes'))

        time.sleep(10)


if __name__ == '__main__':
    main()