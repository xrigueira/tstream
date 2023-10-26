from data_provider.data_factory import data_provider

import argparse

# parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# data loader
args = {
        'data': 'ETTm1',
        'root_path': './data/ETT/',
        'data_path': 'ETTh1.csv',
        'features': 'M',
        'target': 'OT',
        'freq': 'h',
        'checkpoints': './checkpoints/',
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 96,
        'embed': 'timeF',
        'num_workers': 10,
        'batch_size': 32
    }

# parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
# parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
# parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
# parser.add_argument('--features', type=str, default='M',
#                     help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
# parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
# parser.add_argument('--freq', type=str, default='h',
#                     help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
# parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# # forecasting task
# parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
# parser.add_argument('--label_len', type=int, default=48, help='start token length')
# parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# # model define
# parser.add_argument('--embed', type=str, default='timeF',
#                         help='time features encoding, options:[timeF, fixed, learned]')

# # optimization
# parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

# args = parser.parse_args()

if __name__ == '__main__':

    data_set, data_loader = data_provider(args, flag='train')
    print(data_set)