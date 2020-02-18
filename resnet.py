import  torch
from    torch import  nn
from    utils import Flatten
from    torch.nn import functional as F
from    torchvision.models import resnet18

'''
class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        # we add stride support for resbok, which is distinct from tutorials.
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(ch_out)
        )


    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        test_x = self.extra(x)
        
        print('x_shape:', x.shape)
        print('testx_shape:' , test_x.shape)
        print('out_shape:', out.shape)
        
        if out.shape != x.shape:
            out = self.extra(x) + out
        else:
            out = x + out

        out = F.relu(out)

        return out

'''


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        trained_model = resnet18(pretrained=True)
        self.layer1 = nn.Sequential(*list(trained_model.children())[:-1])     # [b, 512, 1, 1]
        self.layer2 = Flatten()
        self.layer3 = nn.Linear(512, 30)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(30, 5)
        '''
        # [b, 16, h, w] => [b, 32, h ,w]
        self.blk1 = ResBlk(16, 32, stride=3)
        # [b, 32, h, w] => [b, 64, h, w]
        self.blk2 = ResBlk(32, 64, stride=2)
        # # [b, 64, h, w] => [b, 128, h, w]
        self.blk3 = ResBlk(64, 128, stride=2)
        # # [b, 128, h, w] => [b, 256, h, w]
        self.blk4 = ResBlk(128, 256, stride=2)

        self.outlayer = nn.Linear(256*4*4, num_class)
        '''

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        '''
        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)


        # print('after conv:', x.shape) #[b, 512, 2, 2]
        # [b, 512, h, w] => [b, 512, 1, 1]
        #x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        '''

        return x




def main():

    tmp = torch.randn(2, 3, 224, 224)
    model = ResNet18()
    out = model(tmp)
    print(model)
    print(out.shape)



if __name__ == '__main__':
    main()