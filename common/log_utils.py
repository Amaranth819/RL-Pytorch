import os
from tensorboardX import SummaryWriter

class Logger(object):
    def __init__(self, log_path : str):
        self.summary = SummaryWriter(log_path)


    def add(self, step, scalar_dict):
        for tag, scalar_val in scalar_dict.items():
            self.summary.add_scalar(tag, scalar_val, step)


        print('####################')
        print('# Epoch: %d' % step)
        for tag, scalar_val in scalar_dict.items():
            print('# {}: {:.4e}'.format(tag, scalar_val))
        print('####################\n')


    def close(self):
        self.summary.close()