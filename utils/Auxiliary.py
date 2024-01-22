from torch.nn import Module
from torch import nn


# 预测器
class Predictor(Module):
    def __init__(self, in_features, hide_features, pred_len):
        super(Predictor, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, hide_features),
            nn.ReLU(),
            nn.Linear(hide_features, pred_len)
        )

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, in_feature)
        :return: (batch_size, seq_len, pred_len * 3)
        """
        x = self.linear(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.opt.in_features) + ' -> ' \
            + str(self.opt.pred_len * 3) + ')'
