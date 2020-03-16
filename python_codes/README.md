# data_analysis
class_moving_series.py:  创建移动特征的类。其中包括: 创建移动相关值、移动平均、移动最小值、移动最大值、移动中间值、移动方差、移动峰值、 移动倾斜度、指数移动平均、指数移动方差、二次指数移动平均。

create_time_feature.py，create_time_feat_one_hot.py：创建与时间相关的特征，后者在前者基础上进行了one_hot处理。

operate_influxdb.py: 操作influxdb数据库。实现数据的的增删查，查询数据返回格式转换为pandas中的dataframe格式。

lstm.py、main_LSTM.py: Long Short Term Mermory network, 用于时序预测的类。

df_format_transform.py：绍兴数据格式的格式转换，将codename:value形式转换为以以codename具体值命名的列，其中利用了数据采集时间作为merge操作时的key。

time_convert.py：时间格式转换小工具。

find_window_size.py：计算用来创建移动特征时的最佳window_size。

correlation between.py：绍兴48hr数据的数据相关性的计算。
