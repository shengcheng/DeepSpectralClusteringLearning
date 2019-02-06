import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import numpy as np



from dataset.data_loader import get_data_loader

from utils.utils import str2bool
from utils.utils import set_devices
from utils.utils import may_set_mode

from models.Model import Model
from models.SpectralClusterLayer import SpectralCLusterLayer

import argparse

class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        # parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--data_dir', type=str, default='')
        parser.add_argument('--is_train', type=str2bool, default=True)

        parser.add_argument('--resize_h_w', type=eval, default=(256, 256))
        parser.add_argument('--crop_h_w', type=eval, default=(227, 227))
        parser.add_argument('--mirror', type=str2bool, default=True)

        parser.add_argument('--ids_per_batch', type=int, default=32)
        parser.add_argument('--ims_per_id', type=int, default=4)

        # parser.add_argument('--num_class', type=int, default=196)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--vector_size', type=int, default=1024)
        parser.add_argument('--fix_weight', type=str2bool, default=False)
        parser.add_argument('--total_epoch', type=int, default=500)
        parser.add_argument('--lr_start ')

        args = parser.parse_args()

        self.sys_device_ids = args.sys_device_ids
        # self.num_workers = args.num_workers
        self.data_dir = args.data_dir
        self.is_train = args.is_train

        self.resize_h_w = args.resize_h_w
        self.crop_h_w = args.crop_h_w
        self.mirror = args.mirror
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]
        self.scale = True

        self.ids_per_batch = args.ids_per_batch
        self.ims_per_id = args.ims_per_id
        self.batch_size = self.ids_per_batch * self.ims_per_id

        # self.num_class = args.num_class
        self.fix_weight = args.fix_weight
        self.vector_size = args.vector_size
        self.lr = args.lr
        self.total_epoch = args.total_epoch

def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):

        assert ep >= 1, "Current epoch number should be >= 1"

        if ep < start_decay_at_ep:
            return

        for g in optimizer.param_groups:
            g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                        / (total_ep + 1 - start_decay_at_ep))))
            print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))



def main():
    cfg = Config()
    TVT, TMO = set_devices(cfg.sys_device_ids)
    data_loader = get_data_loader(cfg)
    spec_loss = SpectralCLusterLayer()
    model = Model(cfg.vector_size, cfg.fix_weight)
    model_w = DataParallel(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = cfg.lr)
    modules_optims = [model, optimizer]
    TMO(modules_optims)

    may_set_mode(modules_optims, 'train')
    for epoch in range(cfg.total_epoch):
        epoch_done = False
        step = 0
        while not epoch_done:
            step += 1
            ims, _, labels, epoch_done= data_loader.next_batch()
            ims_var = Variable(TVT(torch.from_numpy(ims).float()))
            batch_size = ims_var.size()[0]
            num_cluster = len(data_loader.ids)
            labels_matrix = np.zeros([batch_size, num_cluster], dtype=int)
            labels_matrix[range(batch_size), labels-1] = 1
            labels_var = TVT(torch.from_numpy(labels_matrix).float())
            optimizer.zero_grad()
            feat = model_w(ims_var)
            G = spec_loss.grad_F(feat, labels_var)
            feat.backward(gradient=G)
            optimizer.step()
            objective_value = labels_var.size()[1] - torch.sum(torch.mm(spec_loss.pseudo_inverse(labels_var),feat)  * torch.mm(spec_loss.pseudo_inverse(feat), labels_var).t())
            print("epoch %d --- loss value= %f" % (epoch, objective_value))
    print "Finished"




if __name__ == '__main__':
    main()
