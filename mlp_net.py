import torch
from   torch import nn, optim


class MLP_net(nn.Module):
    def __init__(self ):
        super(MLP_net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(36, 12),
            nn.ReLU(),
            # nn.Linear(12, 12),
            # nn.ReLU(),
            nn.Linear(12, 5)
        )
        # tmp = torch.randn(2,16)
        # out = self.model(tmp)

        # print(' out:', out.shape)

    def forward(self, x):

        x = self.model(x)

        return x


'''
def main():

    net = MLP_net()
    print(net)

    
    tmp = torch.randn(32, 16)
    # batchsz = tmp.size(0)
    # tmp = tmp.view(batchsz, -1)
    print(type(tmp), tmp.shape, tmp)
    out = net(tmp)
    print('MLP:', out.shape)

if __name__ == '__main__':
    main()
'''










