 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --use_RevIN False
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --IsLocationEncoder False
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --IsLocationInfo False
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --IsLocationEncoder False --IsLocationInfo False
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --Is_wind_angle False
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --Is_wind_speed False
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --Is_wind_speed False --Is_wind_angle False
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --Is_air_temperature False
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --Is_dew_point False
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --Is_dew_point False --Is_air_temperature False
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --IsDayOfYearEmbedding True
 python train.py --model_des $RANDOM --dataset '1D-data' --num_spatial_att_layer 2 --num_temporal_att_layer 2 --end_channels 1024