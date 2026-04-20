from math import exp
import os
import argparse  
import random
import sys
import yaml
import torch
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)
os.chdir(root_dir)
sys.path.append(os.path.join(root_dir, 'src', 'pipeline'))

from src.pipeline.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from src.pipeline.graphics.plot import PredictionPlotter
from src.pipeline.data_provider.data_factory import data_provider


def load_config(config_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.abspath(os.path.join(current_dir, '..', 'config', config_path))
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Arquivo não encontrado em: {full_path}")
    with open(full_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.set_num_threads(6)

    config           = "config.yaml"
    checkpoint_setting = "output_1h"
    checkpoint_path  = f'./artifacts/checkpoints/{checkpoint_setting}/checkpoint.pth'

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint não encontrado: {checkpoint_path}")

    print(f"✅ Checkpoint encontrado: {checkpoint_path}")

    config_dict = load_config(config)
    args        = argparse.Namespace(**config_dict)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast(args)


    save_time = datetime.now().strftime('%d-%m-%Y_%H_%M-%S')
    folder    = os.path.join(r"C:\Users\Pichau\OneDrive\Desktop\Project-1\KANMTS\imgs", f"Test_{save_time}")
    os.makedirs(folder, exist_ok=True)
    print(f'📁 Gráficos serão salvos em: {folder}')

    train_data, _ = data_provider(args, flag='test')
    print(f"✅ Canais detectados: {train_data.get_channel_names()}")

    plot = PredictionPlotter(model_path=checkpoint_path, denormalize=True, save_path=folder) 
    plot.load_test(args)          
    plot.plot_feature(feature_name="PETR4_OPEN", horizon=0, show_plot=True)
    plot.plot_all_features(horizon=0)

   












