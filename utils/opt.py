#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime
from pprint import pprint

import pytz

from utils import log
import sys


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        # self.parser.add_argument('--data_dir', type=str,
        #                          default='/home/wei/Documents/',
        #                          help='path to dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--test', action='store_true', help='whether to test the model')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether it is to evaluate the model')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--checkpoint_path', type=str, default=None, help='path to checkpoint')
        self.parser.add_argument('--skip_rate', type=int, default=5, help='skip rate of samples')
        self.parser.add_argument('--skip_rate_test', type=int, default=5, help='skip rate of samples for test')

        # ===============================================================
        #                     Model options
        # ===============================================================
        # self.parser.add_argument('--input_size', type=int, default=2048, help='the input size of the neural net')
        # self.parser.add_argument('--output_size', type=int, default=85, help='the output size of the neural net')
        self.parser.add_argument('--in_features', type=int, default=54, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='size of each model layer')
        self.parser.add_argument('--d_model', type=int, default=256, help='past frame number')
        self.parser.add_argument('--kernel_size', type=int, default=5, help='past frame number')
        self.parser.add_argument('--increment', type=bool, default=True, help='use increament learning')
        # self.parser.add_argument('--drop_out', type=float, default=0.5, help='drop out probability')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--input_n', type=int, default=50, help='past frame number')
        self.parser.add_argument('--output_n', type=int, default=25, help='future frame number')
        self.parser.add_argument('--dct_n', type=int, default=10, help='future frame number')
        self.parser.add_argument('--lr_now', type=float, default=0.0003)
        self.parser.add_argument('--max_norm', type=float, default=10000)
        self.parser.add_argument('--epoch', type=int, default=100)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--test_batch_size', type=int, default=16)
        self.parser.add_argument('--is_load', dest='is_load', action='store_true',
                                 help='whether to load existing model')
        self.parser.add_argument('--name', default='train',
                                 help='whether to load existing model')
        # ===============================================================
        #                     TimesNet options
        # ===============================================================
        self.parser.add_argument('--top_k', type=int, default=3, help='top k frequency')
        self.parser.add_argument('--e_layers', type=int, default=2, help='TimesBlock layers')
        self.parser.add_argument('--enc_in', type=int, default=3, help='input channel')
        self.parser.add_argument('--d_ff', type=int, default=32, help='hidden layer size')
        self.parser.add_argument('--num_kernels', type=int, default=6, help='number of kernels')
        self.parser.add_argument('--embed', type=str, default='timeF', help='embedding type')
        self.parser.add_argument('--freq', type=str, default='h', help='frequency')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
        self.parser.add_argument('--c_out', type=int, default=3, help='output channel')
        self.parser.add_argument('--seq_len', type=int, default=50, help='input sequence length')
        self.parser.add_argument('--pred_len', type=int, default=10, help='prediction sequence length')
        # ===============================================================
        #                    TSFormer options
        # ===============================================================
        self.parser.add_argument('--embed_dim', type=int, default=96, help='number of workers')
        self.parser.add_argument('--tsformer', type=str, default=None, help='path to checkpoint')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        if not self.opt.is_eval:
            # 设置时区
            china_timezone = pytz.timezone('Asia/Shanghai')
            current_timestamp = datetime.now(china_timezone).strftime('%H%M%S')
            log_name = self.opt.name + '_' + current_timestamp
            self.opt.exp = log_name
            ckpt = os.path.join(self.opt.ckpt, datetime.now(china_timezone).strftime('%Y%m%d'))
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
            ckpt = os.path.join(ckpt, self.opt.exp)
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
                log.save_options(self.opt)
            self.opt.ckpt = ckpt
            log.save_options(self.opt)
        self._print()
        # log.save_options(self.opt)
        return self.opt
