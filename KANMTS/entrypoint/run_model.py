import os
import argparse  
import random
from re import S
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
    # fix seed for reproducibility
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.set_num_threads(6)

    
    config = "config.yaml"
    
    print(f"📄 Carregando configurações de: {config}")
    config_dict = load_config(config)
    args = argparse.Namespace(**config_dict)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)
    save_time_model = datetime.now().strftime('%d-%m-%Y_%H_%M-%S')
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]

    setting = f'Train_model_test_{dataset_name}_{save_time_model}_{args.seq_len}_{args.patience}'



    print(f'>>>>>>> Iniciando Experimento: {setting} >>>>>>>')
    print(f'>>>>>>> Configuração utilizada: {args} >>>>>>>')
    Exp = Exp_Long_Term_Forecast(args)
    
    model = Exp.train(setting=setting)
    Exp.test(setting=setting, test=1)



     








        






    