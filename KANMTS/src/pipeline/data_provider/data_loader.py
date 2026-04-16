import os
import warnings

import numpy as np
from onnx_ir import val
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class PETR4_dataset(Dataset):
    def __init__(self, root_path, data_path, flag='train', scale=True, timeenc=1, freq='15min', size=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]  
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.root_path = root_path
        self.scale = scale
        self.data_path = data_path
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = RobustScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test



        """
        border1s marca o inicio de cada split de dados
        border1s[0] = 0 -> Define o início da região de treino
        border1s[1] = num_train - seq_len -> Primeiro índice da janela de validação, a subtração por seq_len ocorre porque o modelo precisa "olhar para  tras"
        seq_len passos. Utilizamos para que os seq_len anteriores sirvam de contexto para o modelo validar o treino
        border1s[2] = len(df_raw) - num_test - self.seq_len -> Primeiro índice da janela de testes, subtraimos seq_len para que teste tenha contexto da validação


        border2s definem o fim de cada split de dados
        """

        border1s = [0, num_train - self.seq_len, int(len(df_raw) - num_test - self.seq_len)]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_base = df_raw.columns[1:]
        self.cols_base = list(cols_base)
        print(f'Colunas Base : {cols_base}')
        df_base = df_raw[cols_base]

        if self.scale:
            train_data = df_base[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_base.values)
        else:
            data = df_base.values
        data = data.astype(np.float32)

        df_stamp = df_raw[['DATE']][border1 : border2]
        df_stamp['DATE'] = pd.to_datetime(df_stamp.DATE)

        df_stamp = df_raw[['DATE']][border1:border2]
        df_stamp['DATE'] = pd.to_datetime(df_stamp.DATE)

        if self.timeenc == 0:
            print(f"[DEBUG] timeenc == 0")
            df_stamp['month'] = df_stamp.DATE.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.DATE.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.DATE.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.DATE.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.DATE.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['DATE'], axis= 1).values
        elif self.timeenc == 1: # Utiliza um método de codificação de tempo (Time Embedding)
            print(f"[DEBUG] timeenc == 1")
            data_stamp = time_features(pd.to_datetime(df_stamp['DATE'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            

        data_stamp = np.asarray(data_stamp, dtype=np.float32) # Garantir que o retorno seja um array numpy do tipo float32
        self.data_x = data[border1 : border2]    # São iguais agora pois meu get_items fará a separação depois
        self.data_y = data[border1 : border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin : s_end]
        seq_y = self.data_y[r_begin : r_end]
        seq_x_mark = self.data_stamp[s_begin : s_end]
        seq_y_mark = self.data_stamp[r_begin : r_end]

        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
        
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    def inverse_transform(self, data):
        is_3d = False
        if not self.scale:
            return data
        if len(data.shape) == 3:
            B, T, C = data.shape
            data= data.reshape(-1, C)
            is_3d = True
        data_denorm = self.scaler.inverse_transform(data)
        if is_3d:
            return data_denorm.reshape(B, T, C)
        
        return data_denorm
    def get_channel_names(self):
        return self.cols_base


