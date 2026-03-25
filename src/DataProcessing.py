import logging
import os
import sys
# from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler

from src.utils import Scaler
from src import utils


class RadiationDataProcessing:
    def __init__(self, config):
        self.config = config
        self.noaa_list = []
        if self.config['Is_wind_angle']:
            self.noaa_list.append('wind_angle')
        if self.config['Is_wind_speed']:
            self.noaa_list.append('wind_speed')
        if self.config['Is_air_temperature']:
            self.noaa_list.append('air_temperature')
        if self.config['Is_dew_point']:
            self.noaa_list.append('dew_point')

        self.traffic_data = {}
        self.nodeID = self.read_idx()
        self.adj_mx_01 = self.read_adj_mat()

        self.dataloader = {}
        self.loc_ft = self.read_loc()
        self.dataloader['loc_feature'] = self.loc_ft

        # Iteration 4: Region-aware clustering on station locations
        num_clusters = self.config.get('num_region_clusters', 0)
        if num_clusters > 0:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_clusters, random_state=self.config.get('seed', 2025), n_init=10)
            self.cluster_ids = kmeans.fit_predict(self.loc_ft)  # [N,]
        else:
            self.cluster_ids = None
        self.dataloader['cluster_ids'] = self.cluster_ids

        self.build_data_loader()

    def read_loc(self):
        loc = pd.read_csv(f"{self.config['DATA_PATH']}/{self.config['dataset']}/location_info.csv")
        loc_ft = np.zeros((self.config['num_sensors'], 2))
        for i in range(loc.shape[0]):
            loc_ft[i, :] = loc.iloc[i, 2:4].tolist()
        loc_ft = (loc_ft - loc_ft.mean(axis=0)) / (loc_ft.std(axis=0) + 1e-8)
        return loc_ft

    def build_data_loader(self):
        train_traffic, valid_traffic, test_traffic = self.read_traffic()

        train_noaa, valid_noaa, test_noaa = {}, {}, {}

        if len(self.noaa_list) > 0:
            for name in self.noaa_list:
                # print(self.config['noaa_list'])
                train_noaa[name], valid_noaa[name], test_noaa[name] = self.read_noaa(tag=name)

        train_data = train_traffic[list(self.nodeID.keys())]

        # Iteration 1a: Log-space transform before normalization
        self.use_log_space = self.config.get('use_log_space', False)
        if self.use_log_space:
            # Apply log1p to compress 2-order-of-magnitude range (40-7170 nSv/h)
            # Scaler will then normalize in log-space
            raw_max = train_data.values[train_data.values != 0].max()
            train_data_for_scaler = np.log1p(train_data.values.clip(min=0))
            self.scaler = Scaler(train_data_for_scaler, missing_value=0)
            # Override max/min with original-space values (for clamp after expm1)
            self.scaler.max_value = raw_max
            self.scaler.min_value = 0.0
        else:
            self.scaler = Scaler(train_data.values, missing_value=0)

        # data for training & evaluation
        self.get_data_loader(train_traffic, train_noaa, shuffle=True, tag='train')
        self.get_data_loader(valid_traffic, valid_noaa, shuffle=False, tag='valid')
        self.get_data_loader(test_traffic, test_noaa, shuffle=False, tag='test')

    def get_data_loader(self, data, noaa, shuffle, tag):
        if len(data) == 0:
            return 0
        num_timestamps = data.shape[0]

        data_time = data.iloc[:, 0]
        data_time = pd.to_datetime(data_time, utc=None)
        # data_time = data_time.dt.tz_localize(None)
        self.traffic_data[tag+'_data'] = data

        data = data[list(self.nodeID.keys())]

        # fill missing value
        data_fill = self.fill_traffic(data)

        # transform data distribution
        data_values = data_fill.values
        if self.use_log_space:
            data_values = np.log1p(data_values.clip(min=0))
        in_data = np.expand_dims(self.scaler.transform(data_values), axis=-1)  # [T, N, 1]

        if self.config['IsLocationInfo']:
            if tag == 'train':
                num = self.num_train
            elif tag == 'valid':
                num = self.num_valid
            else:
                num = self.num_test

            location_info = np.repeat(self.loc_ft[np.newaxis, :, :], num, axis=0)
            in_data = np.concatenate([in_data, location_info], axis=-1)  # [T, N, D]


        # time in day
        # if self.config['IsTimeEmbedding']:
        #     time_ft = (pd.to_datetime(data_time.values) - data_time.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        #     time_ft = np.tile(time_ft, [1, self.config['num_sensors'], 1]).transpose((2, 1, 0))  # [T, N, 1]
        #     in_data = np.concatenate([in_data, time_ft], axis=-1)  # [T, N, D]

        # day in month (0-1)
        # if self.config['IsDayEmbedding']:
        #     dt_series = pd.to_datetime(data_time.values).to_series()
        #     days_in_month = dt_series.dt.days_in_month.values
        #     day_of_month = dt_series.dt.day.values
        #     time_ft = (day_of_month - 1) / (days_in_month - 1)
        #     time_ft = np.tile(time_ft, [1, self.config['num_sensors'], 1]).transpose((2, 1, 0))
        #     in_data = np.concatenate([in_data, time_ft], axis=-1)

        # month in year (0-1)
        # if self.config['IsMonthEmbedding']:
        #     month = pd.to_datetime(data_time.values).to_series().dt.month.values
        #     time_ft = (month - 1) / 11
        #     time_ft = np.tile(time_ft, [1, self.config['num_sensors'], 1]).transpose((2, 1, 0))
        #     in_data = np.concatenate([in_data, time_ft], axis=-1)

        if self.config['IsDayOfYearEmbedding']:
            dt_series = pd.to_datetime(data_time.values).to_series()
            # 获取一年中的第几天 (1-366)
            time_ft = dt_series.dt.dayofyear.values-1
            # 统一使用366天归一化
            # time_ft = (time_ft - 1) / 365  # 365 而不是 366-1
            time_ft = np.tile(time_ft, [1, self.config['num_sensors'], 1]).transpose((2, 1, 0))
            in_data = np.concatenate([in_data, time_ft], axis=-1)


        if len(self.noaa_list) > 0:
            for name in self.noaa_list:
                array = noaa[name].values
                # Use training set statistics to normalize all splits (prevent data leakage)
                if not hasattr(self, 'noaa_stats'):
                    self.noaa_stats = {}
                if name not in self.noaa_stats:
                    # First call is always training set - save its statistics
                    self.noaa_stats[name] = {
                        'mean': array.mean(axis=0),
                        'std': array.std(axis=0) + 1e-8
                    }
                normalized_array = (array - self.noaa_stats[name]['mean']) / self.noaa_stats[name]['std']
                d = np.expand_dims(normalized_array, axis=-1)
                in_data = np.concatenate([in_data, d], axis=-1)  # [T, N, D]
        out_data = np.expand_dims(data.values, axis=-1)  # [T, N, 1]

        # create inputs & labels
        inputs, labels = [], []
        for i in range(self.config['in_length']):
            temp = in_data[i: num_timestamps + 1 - self.config['in_length'] - self.config['out_length'] + i]
            inputs += [temp]
        for i in range(self.config['out_length']):
            temp = out_data[self.config['in_length'] + i: num_timestamps + 1 - self.config['out_length'] + i]
            labels += [temp]
        # inputs = np.stack(inputs).transpose((1, 3, 2, 0))
        # labels = np.stack(labels).transpose((1, 3, 2, 0))
        inputs = np.stack(inputs).transpose((1, 0, 2, 3))
        labels = np.stack(labels).transpose((1, 0, 2, 3))

        # logging info of inputs & labels
        logging.info('load %s inputs & labels [ok]', tag)
        logging.info('input shape: %s', inputs.shape)  # [num_timestamps, c, n, input_len]
        logging.info('label shape: %s', labels.shape)  # [num_timestamps, c, n, output_len]

        # create dataset
        # dataset = TensorDataset(
        #     torch.from_numpy(inputs).to(dtype=torch.float),
        #     torch.from_numpy(labels).to(dtype=torch.float)
        # )
        # create sampler
        # sampler = SequentialSampler(dataset)
        # if shuffle:
        #     sampler = RandomSampler(dataset, replacement=True, num_samples=self.config['batch_size'])
        # else:
        #     sampler = SequentialSampler(dataset)
        # create dataloader
        # data_loader = DataLoader(dataset=dataset, batch_size=self.config['batch_size'], sampler=sampler,
        #                          num_workers=4, drop_last=False)

        self.dataloader[tag+'_loader'] = DataLoaderM(inputs, labels, self.config['batch_size'])
        self.dataloader['x_'+tag] = inputs
        self.dataloader['y_'+tag] = labels

        return None

    def read_idx(self):
        with open(os.path.join(self.config['DATA_PATH'], self.config['dataset'], 'node_id.txt'), mode='r', encoding='utf-8') as f:
            ids = f.read().strip().split('\n')
        idx = {}
        for i, sensor_id in enumerate(ids):
            idx[sensor_id] = i
        return idx

    def read_adj_mat(self):
        # 更改 邻接矩阵只保留距离信息
        graph_csv = pd.read_csv(f"{self.config['DATA_PATH']}/{self.config['dataset']}/node_distance_{self.config['distance']}.csv",
                                dtype={'from': 'str', 'to': 'str'})

        # 0, 1 adjacency matrix
        adj_mx_01 = np.zeros((self.config['num_sensors'], self.config['num_sensors']))
        for k in range(self.config['num_sensors']):
            adj_mx_01[k, k] = 1

        for row in graph_csv.values:
            if row[0] in self.nodeID and row[1] in self.nodeID:
                # 01 adjacency matrix
                adj_mx_01[self.nodeID[row[0]], self.nodeID[row[1]]] = 1  # 0, 1
        return adj_mx_01


    def read_noaa(self, tag):
        data = pd.read_csv(f"{self.config['DATA_PATH']}/{self.config['dataset']}/noaa/{tag}.csv")

        num_train = int(data.shape[0] * self.config['train_prop'])
        num_valid = int(data.shape[0] * self.config['valid_prop'])
        num_test = data.shape[0] - num_train - num_valid

        train = data[:num_train].copy()
        valid = data[num_train: num_train + num_valid].copy()
        test = data[-num_test:].copy()

        return train, valid, test


    def read_traffic(self):
        data = pd.read_csv(f"{self.config['DATA_PATH']}/{self.config['dataset']}/data.csv")
        # self.data_time = data.iloc[:, 0]

        self.num_train = int(data.shape[0] * self.config['train_prop'])
        self.num_valid = int(data.shape[0] * self.config['valid_prop'])
        self.num_test = data.shape[0] - self.num_train - self.num_valid

        train = data[:self.num_train].copy()
        valid = data[self.num_train: self.num_train + self.num_valid].copy()
        test = data[-self.num_test:].copy()

        return train, valid, test

    def fill_traffic(self, data):
        data = data.copy()
        # Only treat exact zeros as missing (sensor outage), not low legitimate readings
        data[data == 0] = float('nan')
        data = data.ffill()
        data = data.bfill()
        return data


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()