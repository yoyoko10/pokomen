import  torch
import  os, glob
import  random, csv
import  numpy as np
from    torch.utils.data import Dataset, DataLoader
from    torchvision import transforms
from    PIL import Image


class Pokemon(Dataset):

    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {} # "sq...":0  这里定义了一个类
        for name in sorted(os.listdir(os.path.join(root))):   # 按顺序遍历根目录下面的文件夹
            if not os.path.isdir(os.path.join(root, name)):   # 判断如果该name不是文件夹
                continue     # 这里是指跳过非文件夹存储图片的其他类型文件

            # 这里就拿到了文件夹的名字
            self.name2label[name] = len(self.name2label.keys())
            # 返回name的个数作为文件夹的标签

        print(self.name2label)

        # image, label
        self.images, self.labels, self.atts = self.load_csv('images.csv')

        if mode=='train': # 60%
            self.images = self.images[:int(0.4*len(self.images))]
            self.labels = self.labels[:int(0.4*len(self.labels))]
            self.atts = self.atts[:int(0.4 * len(self.atts))]

            #60%的数据用于train
        elif mode=='val': # 20% = 60%->80%
            self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.atts = self.atts[int(0.6 * len(self.atts)):int(0.8 * len(self.atts))]
            self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
            #60%-80%用于validation

        elif mode == 'extract':
            self.images = self.images
            self.atts = self.atts
            self.labels = self.labels

        else: # 20% = 80%->100%
            self.images = self.images[int(0.8*len(self.images)):]
            self.atts = self.atts[int(0.8 * len(self.atts)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]
            #80%-100%用于test





    def load_csv(self, filename):
        # 如果 filename这个文件不存在，就开始创建添加一个
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []       # 建立一个images的列表，里面的元素是每个图片的路径
            for name in self.name2label.keys():
                # 把每个name下的文件都保存下来
                # 'pokemon\\mewtwo\\00001.png
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            # 1167, 'pokemon\\bulbasaur\\00000000.png'
            print(len(images), images)

            random.shuffle(images)
            # 创建一个命名为“filename”的csv文件
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2]   # name = bulbasaur
                    label = self.name2label[name]    # 获取name对应的标签

                    # 加入其他属性特征> PH > RES > AGI > ATK > MGK > LUK
                    if label == 0:
                        PH, SPEED = random.uniform(3, 5), random.uniform(2, 4)
                        ATT, DEF = random.uniform(3, 5), random.uniform(4, 6)
                        S_ATT, S_DEF = random.uniform(3, 5), random.uniform(7, 9)

                    elif label == 1:
                        PH, SPEED = random.uniform(7, 9), random.uniform(1, 3)
                        ATT, DEF = random.uniform(5, 7), random.uniform(6, 8)
                        S_ATT, S_DEF = random.uniform(2, 4), random.uniform(5, 7)
                    elif label == 2:
                        PH, SPEED = random.uniform(2, 4), random.uniform(3, 5)
                        ATT, DEF = random.uniform(4, 6), random.uniform(2, 4)
                        S_ATT, S_DEF = random.uniform(9, 10), random.uniform(3, 5)
                    elif label == 3:
                        PH, SPEED = random.uniform(3, 5), random.uniform(6, 8)
                        ATT, DEF = random.uniform(4, 6), random.uniform(3, 5)
                        S_ATT, S_DEF = random.uniform(7, 9), random.uniform(3, 5)
                    else:
                        PH, SPEED = random.uniform(5, 7), random.uniform(1, 3)
                        ATT, DEF = random.uniform(3, 5), random.uniform(9, 10)
                        S_ATT, S_DEF = random.uniform(3, 5), random.uniform(3, 5)


                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label, PH, SPEED, ATT, DEF, S_ATT, S_DEF])   # 将图片的文件路径和对应的类型标签存贮到csv文件中
                print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        atts = []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0, [att]
                img = row[0]
                label = row[1]
                att = row[2:]
                label = int(label)
                att = np.asarray(att)              # 先将列表转换为矩阵形式，矩阵内元素这时还是str类型
                att = att.astype(np.float)        # 将str转换为float才能再变为tensor
                images.append(img)
                labels.append(label)
                atts.append(att)


        assert len(images) == len(labels) == len(atts)

        return images, labels, atts


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
        img, label, att= self.images[idx], self.labels[idx], self.atts[idx]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path= > image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)
        att = torch.tensor(att, dtype=torch.float32)
        # att = att.unsqueeze(0)


        return img, label, att





def main():

    import  visdom
    import  time
    import  torchvision

    viz = visdom.Visdom()

    # tf = transforms.Compose([
    #                 transforms.Resize((64,64)),
    #                 transforms.ToTensor(),
    # ])
    # db = torchvision.datasets.ImageFolder(root='pokemon', transform=tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # print(db.class_to_idx)
    #
    # for x,y in loader:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #
    #     time.sleep(10)


    db = Pokemon('pokemon', 224, 'test')

    x, y, z = next(iter(db))

    print(z, type(z))
    # y = str.split()
    print(z)

    print('sample:', x.shape, y.shape, z.shape, x, y, z)

    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)

    for x, y, z in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(z.numpy()), win='attibutes', opts=dict(title='batch_attributes'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-label'))

        time.sleep(10)
'''
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)

    for x, y  in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-label'))

        time.sleep(10)
'''


if __name__ == '__main__':
    main()