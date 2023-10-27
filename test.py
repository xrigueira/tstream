import warnings
warnings.filterwarnings('ignore')

from data_provider.data_loader import data_provider

# Define arguments
args = {
        'data': 'weather',
        'root_path': './data/weather/',
        'data_path': 'weather.csv',
        'features': 'M',
        'target': 'OT',
        'freq': '10min',
        'checkpoints': './checkpoints/',
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 96,
        'embed': 'timeF',
        'num_workers': 8,
        'batch_size': 32
    }

if __name__ == '__main__':

    data_set, data_loader = data_provider(args, flag='train')
    print(data_set)