import torch
import numpy as np
import argparse
import math, yaml
from src import utils
from src.trainer import Trainer
from src.DataProcessing import RadiationDataProcessing, DataLoaderM
import os, sys, random
import pandas as pd
from tqdm import tqdm
from src.model.NRFormer import NRFormer
from src.model.NRFormer_Plus_v1 import PARF
from src.model.NRFormer_Plus_v2 import PGRT
from src.model.NRFormer_Plus_v3 import PGRT

import wandb, datetime

os.environ['WANDB_MODE'] = 'offline'
wandb.login()

import warnings
warnings.filterwarnings('ignore')


def data_matrix_processing(data_matrix):
    data = []
    length = len(data_matrix)
    for i in range(length):
        if i % all_args['out_length'] == 0:
            data.append(data_matrix[i, :, :])
    matrix = torch.cat(data, dim=1)
    all_data = matrix.T

    return all_data


def save_data(matrix_y, matrix_pre, tag):
    # np.savetxt(save_test_folder+'/'+args.dataset+'_'+tag+'.csv', matrix.cpu().detach().numpy(), fmt='%.2f', delimiter=',')
    data_y = pd.DataFrame(matrix_y.cpu().detach().numpy())
    data_y.columns = ['{}-y'.format(a) for a in TFdata.nodeID.keys()]

    data_pre = pd.DataFrame(matrix_pre.cpu().detach().numpy())
    data_pre.columns =['{}-pre-{}'.format(a, args.model_name) for a in TFdata.nodeID.keys()]

    data_csv_temp = pd.concat([data_y, data_pre], axis=1)
    title = list(data_csv_temp.keys())
    title.sort()
    data_csv = data_csv_temp[title]

    original_data = TFdata.traffic_data[tag+'_data']

    data_time = original_data.iloc[all_args['out_length']:len(data_csv)+all_args['out_length'], 0]
    data_csv.insert(loc=0, column='time', value=data_time.tolist())

    data_pre.to_csv(save_test_folder+'/'+args.model_name+'_'+args.dataset+'_'+args.model_des+'_'+tag+'_yhat.csv', index=False)
    data_csv.to_csv(save_test_folder+'/'+args.model_name+'_'+args.dataset+'_'+args.model_des+'_'+tag+'_prediction.csv', index=False)
    original_data.to_csv(save_test_folder+'/'+args.model_name+'_'+args.dataset+'_'+args.model_des+'_'+tag+'_original.csv', index=False)

    print('prediction on [{} dataset] save successful \n'.format(tag))


def predict_model(tag):
    outputs = []
    true_y = torch.Tensor(dataloader['y_{}'.format(tag)]).to(device)
    true_y = true_y.transpose(1, 3)
    true_y = true_y[:, 0, :, :]
    num_test_iteration = math.ceil(true_y.shape[0] / all_args["batch_size"])
    for iter, (x, y) in tqdm(enumerate(dataloader['{}_loader'.format(tag)].get_iterator()), total=num_test_iteration):
        x = torch.Tensor(x).to(device)
        x = x.transpose(1, 3)
        with torch.no_grad():
            prediction = engine.model(x, dataloader['loc_feature'])
            if isinstance(prediction, tuple):
                prediction = prediction[0]
        outputs.append(prediction.squeeze(dim=1))
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:true_y.size(0), ...]
    pred = scaler.inverse_transform(yhat)
    pred = torch.clamp(pred, min=scaler.min_value, max=scaler.max_value)
    mae, mape, rmse = utils.metric(pred, true_y)

    logs = '{} MAE: {:.4f}, {} MAPE: {:.4f}, {} RMSE: {:.4f}'
    print(logs.format(tag, mae, tag, mape, tag, rmse))
    result_file.write(logs.format(tag, mae, tag, mape, tag, rmse)+'\n')

    single_pred = {}
    for i in range(all_args['num_sensors']):
        y_one = true_y[:, i:i + 1, ...].squeeze(dim=1)
        pred_one = pred[:, i:i + 1, ...].squeeze(dim=1)
        mae_one, mape_one, rmse_one = utils.metric(pred_one, y_one)
        k = [k for k, v in TFdata.nodeID.items() if v == i][0]
        single_pred[k] = mae_one

    single_pred = sorted(single_pred.items(), key=lambda x: x[1])
    single_pred_path = save_test_folder + '/{}_single_pred.txt'.format(tag)
    with open(single_pred_path, 'w') as f:
        for k, v in single_pred:
            f.write(str(k) + ': ' + str(v) + '\n')

    data_y = data_matrix_processing(true_y)
    data_pre = data_matrix_processing(pred)
    save_data(data_y, data_pre, tag)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: 0; use -1 for CPU')
    parser.add_argument('--dataset', type=str, default='1D-data', help='model dataset')
    parser.add_argument('--model_name', type=str, default='NRFormer_Plus_v3', help='NRFormer, NRFormer_Plus_v1, v2, v3')

    parser.add_argument('--sudden_change', type=bool, default=False)

    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to search')
    parser.add_argument('--run_times', type=int, default=1, help='number of run')
    parser.add_argument('--model_des', type=str, default='123', help='save model param')
    args = parser.parse_args()

    with open(f'./model_settings/{args.model_name}.yaml', 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
    model_args = hyperparams['model']
    data_args = hyperparams['data']
    trainer_args = hyperparams['trainer']
    if torch.cuda.is_available():
        trainer_args['gpu_count'] = torch.cuda.device_count()
        trainer_args['gpu_info'] = torch.cuda.get_device_name()
    else:
        trainer_args['gpu_count'] = 0
        trainer_args['gpu_info'] = None

    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    model_args['device'] = device

    all_args = {**vars(args), **model_args, **data_args, **trainer_args}

    # data processing
    TFdata = RadiationDataProcessing(all_args)
    scaler = TFdata.scaler
    dataloader = TFdata.dataloader
    mask_support_adj = [torch.tensor(i).to(device) for i in TFdata.adj_mx_01]

    if args.sudden_change:
        value_sameple_list = []
        test_samples = dataloader['x_test'].shape[0]
        x_value = scaler.inverse_transform(dataloader['x_test'])
        for i in range(test_samples):
            mean_y = dataloader['y_test'][i][:, :, 0].mean(axis=1)
            mean_x = x_value[i][:, :, 0].mean(axis=1)
            max_value = mean_y.max()
            min_value = mean_x.min()
            v = max_value - min_value
            value_sameple_list.append(v)
        temp_list = value_sameple_list
        temp_list.sort(reverse=True)

        yuzhi = temp_list[int(test_samples*0.1)]
        indices = [index for index, value in enumerate(value_sameple_list) if value < yuzhi]

        sudden_change_x = np.delete(dataloader['x_test'], indices, axis=0)
        sudden_change_y = np.delete(dataloader['y_test'], indices, axis=0)

        dataloader['sudden_change_x'] = sudden_change_x
        dataloader['sudden_change_y'] = sudden_change_y
        dataloader['sudden_change_loader'] = DataLoaderM(sudden_change_x, sudden_change_y, all_args['batch_size'])


    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if args.model_name == 'NRFormer':
        model = NRFormer(all_args, mask_support_adj)
    elif args.model_name == 'NRFormer_Plus_v1':
        model = PARF(all_args)
    elif args.model_name == 'NRFormer_Plus_v2':
        model = PGRT(all_args)
    elif args.model_name == 'NRFormer_Plus_v3':
        model = PGRT(all_args, mask_support_adj)

    num_Params = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', num_Params)

    save_folder = './logs/' + args.model_name + '/' + args.dataset
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_model_path = os.path.join(save_folder, args.model_name + '_best_train_model_' + str(args.model_des) + '.pt')

    engine = Trainer(model, all_args, scaler, device)
    try:
        best_epoch = engine.load(save_model_path)
        print('\nload architecture [best epoch {}] from {} [done] \n'.format(best_epoch, save_model_path))
    except Exception as e:
        print(e)
        print('load architecture [fail]')
        sys.exit()

    save_test_folder = './model_test/' + args.model_name + '/' + args.dataset + '/' + args.model_des
    if not os.path.exists(save_test_folder):
        os.makedirs(save_test_folder)
    result_path = save_test_folder + '/results.txt'
    result_file = open(result_path, 'w')

    if args.sudden_change:
        outputs = []
        true_test_y = []
        for iter, (x, y) in enumerate(dataloader['sudden_change_loader'].get_iterator()):
            test_x = torch.Tensor(x).to(device)
            test_x = test_x.transpose(1, 3)
            test_y = torch.Tensor(y).to(device)
            test_y = test_y.transpose(1, 3)[:, 0, :, :]

            with torch.no_grad():
                # input (64, 2, 207, 12)
                # test_x = test_x.squeeze(dim=1)
                # test_x = test_x.transpose(1, 2)
                preds = engine.model(test_x, dataloader['loc_feature'])
                if isinstance(preds, tuple):
                    preds = preds[0]

            outputs.append(preds.squeeze(dim=1))
            true_test_y.append(test_y)
        test_yhat = torch.cat(outputs, dim=0)
        true_test_y = torch.cat(true_test_y, dim=0)
        # test_yhat = test_yhat[:test_y.size(0), ...]
        test_pred = scaler.inverse_transform(test_yhat)
        test_pred = torch.clamp(test_pred, min=scaler.min_value, max=scaler.max_value)
        test_mae, test_mape, test_rmse = utils.metric(test_pred, true_test_y)

        log = '{} Average Performance on Test Data - Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f} \n'
        print(log.format(all_args['out_length'], test_mae, test_mape, test_rmse))

        print('Single steps test:')
        mae = []
        mape = []
        rmse = []
        for i in [6, 9, 12, 24]:
            i = i - 1
            pred_singlestep = test_pred[:, :, i]
            real = true_test_y[:, :, i]
            metrics_single = utils.metric(pred_singlestep, real)
            log = 'horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, metrics_single[0], metrics_single[1], metrics_single[2]))
            mae.append(metrics_single[0])
            mape.append(metrics_single[1])
            rmse.append(metrics_single[2])

        print('\nAverage steps test:')
        mae_avg = []
        mape_avg = []
        rmse_avg = []
        for i in [6, 9, 12, 24]:
            pred_avg_step = test_pred[:, :, :i]
            real = true_test_y[:, :, :i]
            metrics_avg = utils.metric(pred_avg_step, real)
            log = 'average {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i, metrics_avg[0], metrics_avg[1], metrics_avg[2]))
            mae_avg.append(metrics_avg[0])
            mape_avg.append(metrics_avg[1])
            rmse_avg.append(metrics_avg[2])


    else:
        data_pool = ['test']
        for name in data_pool:
            predict_model(name)

    print('Great, {} model test have done!'.format(args.model_des))

