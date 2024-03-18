'''
Created on 22.05.2019

@author: florian
'''
import torch


def main():
    i = torch.tensor([[0, 1, 1],
                      [2, 0, 2]])

    v = torch.tensor([3, 4, 5])
    T = torch.sparse.FloatTensor(i, v, torch.Size([2, 3])).to_dense()
    print(T)


if __name__ == '__main__':
    main()
