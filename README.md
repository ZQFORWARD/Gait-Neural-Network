# Gait-Neural-Network
Gait Neutal Network for Human-Exoskeleton Interaction
In order to adapt to the input of the model, it is recommended to build the data set according to the requirements.

The data set format is as follows:
sample 1: channel_1, channels_2, ......, channels_n, inter_channels_1, ... inter_channels_n, lable_data_1, label_data_2, ......, label_data_n, motion_label;
sample 2: channel_1, channels_2, ......, channels_n, inter_channels_1, ... inter_channels_n, lable_data_1, label_data_2, ......, label_data_n, motion_label;
...
sample n: channel_1, channels_2, ......, channels_n, inter_channels_1, ... inter_channels_n, lable_data_1, label_data_2, ......, label_data_n, motion_label;

where, channel_n means your total sensor data channels (channel_n = sensor channels * input dataframe), inter_channels_n means the intermediate data (total_length = intermediate dataframe * sensor channels), 
the label data means the predicted sensor data (data_length = sensor channels * 1), and the motion label is the action that match this data.


In addition, the input dataframe length and intermediate dataframe length can be changed if you needed.
