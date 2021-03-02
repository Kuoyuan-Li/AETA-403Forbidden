For chinese documents please see [README.zh.md](./README.zh.md)

中文版文档请见 [README.zh.md](./README.zh.md)

Open powershell or cmd under `baseline` folder.

unzip the training data and put all csv files into the ‘data’ folder;
Move the `eqlst.csv` 和 `StationInfo.csv` to data folder.

Run mergeData.py to generate pkl file under ‘data’ folder.

Run readData.py to calculate features. The features file will be generated under `area_feature` folder.

Run lgb.py to generate models.

Replace the token in pred.py by your token. then run pred.py to print the prediction.

This baseline will give a basic model. In addition, the way of how to get data and update result by token is also given.
We divide the target area into 8 small areas, the stations in one area will be considered as a group. A time window will slid on
average_sound and average_magn feature of this group to get some statistical characteristics, such as the max, min,and mean of
a week (day granularity).
For the label, we choose earthquakes of the next week (from next monday to next sunday).
Note for you: this Sunday won't be included because we can't get its data when we do prediction.
The prediction should be updated on Sunday (Chinese standard time: UTC+8). Area which have the max magnitude will be used as the final result, and its
center will be the predicted epicenter.
