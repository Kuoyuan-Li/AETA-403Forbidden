在 baseline 目录下打开 cmd 或者 powershell

解压训练数据，将所有 csv 文件置入 data 文件夹；
将 `eqlst.csv` 和 `StationInfo.csv` 置入 data 文件夹

运行 mergeData.py,以生成合并了的 pkl 文件，pkl 文件会储存在 data 文件夹下

运行 readData.py,以计算所需特征；生成的特征储存在 area_feature 文件夹下

运行 lgb.py，按区域计算 model.

将 pred.py 内的 token 替换为你的 token，运行 pred.py,打印出模型预测

该 baseline 意在提供一个基础的预测方式，并演示如何使用 token 获取数据及提交结果。
该模型将预测区域划分为 8 个区域，每个区域内的站点视为一个整体。在计算特征时，采用一个长度可变的滑窗以可变的步长
在站点集的 average_sound 和 average_magn 上滑动，在得到的滑窗数据上计算一系列统计特征，如七日内给定区域内的最大值(日粒度)
的均值、最大值、最小值、最大最小值差等。
label 方面，采用台站集合相应区域内下一周的地震事件(注意，比赛时，结果在周日提交，
周日当日的数据无法得到)；预测时，对比各区域预测的震级大小，取最大震级为预测结果，相应区域的中心作为预测震中。
