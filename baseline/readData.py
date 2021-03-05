from pandas.core.indexing import is_label_like
from typing_extensions import final
from config import *
from toolkit import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# parameter: magn/sound data within specific time period (depends on train/valid) and area, window=7, step=7, 'magn'/'sound', earthquake data in this area, train/valid
def cacu_features(df:pd.DataFrame, window:int, step:int,tag:str,eqData:pd.DataFrame,flag:str) ->pd.DataFrame:
    '''
    生成训练数据。对给定区域，以window(单位：秒)为窗口长生成特征
    '''
    if(len(df)==0):  # Case: no magn or sound data
        return None
    df.reset_index(drop=True,inplace=True)  # reset index of the dataframe to default integer index, remove identical rows
    averageName = tag+'@abs_mean'  # tag: magn or sound
    df.rename(columns={averageName:'average'},inplace=True)  # rename column magn@abs_mean/sound@abs_mean to average
    df['average'] = df['average']-df['average'].mean()  # new average (of each row) = old average - mean of the old average
    df['diff_1'] = df.groupby('StationID')['average'].shift(1)
    df['diff_1'] = df['average'].values - df['diff_1'].values  # new column diff_1, the difference between the average this time and the averge last time for each station

    df.loc[:,'Day'] = df['TimeStamp']
    _start_timestamp = string2stamp(Time_Range[flag][0])  # transfer the start time of train/valid into timestamp
    df['Day'] = df['Day'] - _start_timestamp  # time interval from the start timestamp to the detecting date (in timestamp)
    df['Day'] = (df['Day']//86400+1).astype(int)  # time interval from the start timestamp to the detecting date (in day)
    df.reset_index()  # get rid of identical rows
    tmp = pd.DataFrame(sorted(df['Day'].unique()))  # sort the day
    tmp.columns=['Day']
    res_df = pd.DataFrame((tmp['Day']//step+1).unique()).astype(int)  # res_df: store days//step(week?)+1, remove identical
    res_df.columns=['Day']
    res_df['Day'] = res_df['Day']*step  # convert back to days, with a jump of 7 days(step length)
    for feature in ['average', 'diff_1']:
        for tagging in ['max', 'min', 'mean']:
            kk = df.groupby('Day')[feature].agg(tagging)  #group the data in DF by day (group stations), get the average/diff_1 column and find the max/min/mean data of all records (stations) on that day
            kk.rename(f'{feature}_day_{tagging}',inplace=True)  # rename the column to average_day_max, average_day_min, average_day_mean, diff_1_day_max,diff_1_day_min,diff_1_day_mean
            tmp = pd.merge(tmp, kk, how='left',on='Day') # merge day and the kk data
        #max_mean.min_mean:
        tmp[f'{feature}_day_max_mean'] = tmp[f'{feature}_day_max'].rolling(window=window,center=False).mean()  # rolling window calculation to get the average_day_max_mean/diff_1_day_max_mean
        tmp[f'{feature}_day_min_mean'] = tmp[f'{feature}_day_min'].rolling(window=window,center=False).mean()  # rolling window calculation to get the average_day_min_mean/diff_1_day_min_mean
        #mean_max,mean_min:
        tmp[f'{feature}_day_mean_max'] = tmp[f'{feature}_day_mean'].rolling(window=window,center=False).max()
        tmp[f'{feature}_day_mean_min'] = tmp[f'{feature}_day_mean'].rolling(window=window,center=False).min()
        res_df = pd.merge(res_df,tmp[['Day',f'{feature}_day_max_mean',f'{feature}_day_min_mean',f'{feature}_day_mean_max',f'{feature}_day_mean_min']],on='Day',how='left')
        # merge 4 tmp data columns by res_df days (7 day gap)

        res_df[f'{feature}_mean'] = None
        res_df[f'{feature}_max'] = None
        res_df[f'{feature}_min'] = None
        res_df[f'{feature}_max_min'] = None
        for i,row in res_df.iterrows():  # iterate all days in the res_df
            endDay = row['Day']  # end day is the day on the res_df
            startDay = endDay - window  # start day is 7 days before end day
            data_se = df[(df['Day']>startDay)&(df['Day']<=endDay)][feature]  # df average/diff_1 within the start day and end day
            res_df[f'{feature}_mean'].iloc[i] = data_se.mean()  # the mean of the average/diff_1 in 7 days of all stations with data
            res_df[f'{feature}_max'].iloc[i] = data_se.max()  # the max of the average/diff_1 in 7 days of all stations with data
            res_df[f'{feature}_min'].iloc[i] = data_se.min()  # the min of the average/diff_1 in 7 days of all stations with data
            res_df[f'{feature}_max_min'].iloc[i] = data_se.max() - data_se.min()  # the difference between max and min of the average/diff_1 in 7 days of all stations with data


        res_df[f'{feature}_lastday_mean'] = None
        res_df[f'{feature}_lastday_max'] = None
        res_df[f'{feature}_lastday_min'] = None
        res_df[f'{feature}_lastday_max_min'] = None
        for i,row in res_df.iterrows():
            endDay = row['Day']
            data_last = df[df['Day']==endDay][feature]  # df average/diff_1 data which recorded on the last day of the 7 days
            res_df[f'{feature}_lastday_mean'].iloc[i] = data_last.mean()  # mean of last day data (mean of all stations)
            res_df[f'{feature}_lastday_max'].iloc[i] = data_last.max() # max of last day data (mean of all stations)
            res_df[f'{feature}_lastday_min'].iloc[i] = data_last.min() # min of last day data (mean of all stations)
            res_df[f'{feature}_lastday_max_min'].iloc[i] = data_last.max() - data_last.min()  # max and min difference of last day data (mean of all stations)
    
    for name in res_df.columns.to_list():
        if(name=='Day'):continue
        res_df.rename(columns={name:(name+'_'+tag)},inplace=True)  # rename columns to add tag (magn/sound)
    res_df.dropna(axis=0,how='any',inplace=True)  # if there is any missing data in a row, drop it
    res_df.reset_index(drop=True,inplace=True)  # reset index
    res_df['label_M'] = None
    res_df['label_long'] = None
    res_df['label_lati'] = None
    #起始时间戳
    zero_stamp = _start_timestamp
    for i,row in res_df.iterrows():
        endDay = row['Day']
        endStamp = zero_stamp + (endDay-1)*86400  # convert endday to timestamp
        pre_Range_left = endStamp+86400*2   
        pre_Range_right = endStamp+86400*9   #左闭右开区间,注意，数据窗口和预测周之间隔着周日，周日数据无法即时取得
        #使用时间范围内，区域范围内最大震级的地震作为label 
        _eq = eqData[(eqData['Timestamp']<pre_Range_right) & (eqData['Timestamp']>=pre_Range_left)]  # earthquake data within the timestamp and area
        if(len(_eq)==0):  # no earthquake
            res_df['label_M'].iloc[i] = 0
            res_df['label_long'].iloc[i] = -1
            res_df['label_lati'].iloc[i] = -1
        else:  # earthquake within the area and time
            _eq_max = _eq.iloc[_eq['Magnitude'].argmax()]
            res_df['label_M'].iloc[i] = _eq_max['Magnitude']  # label_M: magnitude
            res_df['label_long'].iloc[i] = _eq_max['Longitude']  # label_long: longitude
            res_df['label_lati'].iloc[i] = _eq_max['Latitude']  # label_lati: latitude
    return res_df  # return the res_df data

    
if __name__ == "__main__":
    EqData = pd.read_csv(Eq_list_path)  # read data/eqlst.csv 地震目录文件
    magn_data = load_object(Merged_Data_Path['magn'])  # read data/magn_data.pkl
    sound_data = load_object(Merged_Data_Path['sound'])  # read data/sound_data.pkl
    area_groups = [  # divide the AETA into 8 groups for 8 areas, set: station IDs, range: latitude: from(0), to(1); longtitude: from(2), to(3)
        {'id':set([133, 246, 119, 122, 59, 127]),'range':[30,34,98,101]},
        {'id':set([128, 129, 19, 26, 159, 167, 170, 182, 310, 184, 188, 189, 191, 197, 201, 204, 88, 90, 91, 93, 94, 221, 223, 98, 107, 235, 236, 252, 250, 124, 125]),'range':[30,34,101,104]},
        {'id':set([141, 150, 166, 169, 43, 172, 183, 198, 202, 60241, 212, 214, 99, 228, 238, 115, 116, 121, 251]),'range':[30,34,104,107]},
        {'id':set([131, 36, 164, 165, 231, 60139, 174, 175, 206, 303, 82, 51, 243, 55, 308, 119, 313, 318]),'range':[26,30,98,101]},
        {'id':set([256, 130, 132, 147, 148, 149, 151, 153, 32, 33, 35, 60195, 38, 39, 41, 302, 304, 177, 305, 307, 181, 309, 314, 315, 316, 317, 319, 320, 193, 322, 200, 73, 329, 75, 333, 78, 334, 84, 87, 60251, 96, 225, 101, 229, 105, 109, 40047, 240, 247, 120, 254, 255]),'range':[26,30,101,104]},
        {'id':set([352, 321, 355, 324, 326, 328, 331, 77, 47, 48, 335, 339]),'range':[26,30,104,107]},
        {'id':set([161, 226, 137, 138, 171, 140, 113, 306, 152, 186, 220, 60157]),'range':[22,26,98,101]},
        {'id':set([50117, 327, 106, 332, 142, 146, 24, 155, 156, 29]),'range':[22,26,101,104]}
    ]

    for i,area in enumerate(area_groups):  # enumerate will count the iteration (i). Iterate each area in the groups
        ID_list = area['id']  # id list
        range_list = area['range']  # range list
        eqData_area = EqData[(EqData['Latitude']>=range_list[0]) & (EqData['Latitude']<=range_list[1]) & 
                                (EqData['Longitude']>=range_list[2]) & (EqData['Longitude']<=range_list[3])]  # earthquake data in this area

        local_magn_data = magn_data[magn_data['StationID'].apply(lambda x:x in ID_list)].reset_index(drop=True)  # magn data of the stations whose IDs are in the list
        local_sound_data = sound_data[sound_data['StationID'].apply(lambda x:x in ID_list)].reset_index(drop=True)  # sound data of the stations whose IDs are in the list
        for flag in ['train', 'valid']:  # train and valid the data
            time_range = Time_Range[flag]  # time range of train or valid
            start_stamp = string2stamp(time_range[0])  # start time of train or valid
            end_stamp = string2stamp(time_range[1])  # end time of train or valid
            _df_magn = local_magn_data[(local_magn_data['TimeStamp']>=start_stamp)&(local_magn_data['TimeStamp']<end_stamp)]  # dataframe to store the magn in the train/valid time
            _df_sound = local_sound_data[(local_sound_data['TimeStamp']>=start_stamp)&(local_sound_data['TimeStamp']<end_stamp)]  # dataframe to store the sound in the train/valid time
            _magn_res = cacu_features(_df_magn,Window,Step,'magn',eqData_area,flag)  # parameter: magn data within specific time period and area, window=7, step=7, 'magn', earthquake data in this area, train/valid
            _sound_res = cacu_features(_df_sound,Window,Step,'sound',eqData_area,flag)  # parameter: sound data within specific time period and area, window=7, step=7, 'sound', earthquake data in this area, train/valid
            #抛弃重合的label列/ drop surplus column
            _magn_res.drop(['label_M','label_long','label_lati'],axis=1,inplace=True)  # drop columns ['label_M','label_long','label_lati'] to avoid redundant
            _final_res = pd.merge(_magn_res,_sound_res,on='Day',how='left')  # merge magn and sound data by day
            _final_res.dropna(inplace=True)  # drop rows with any missing data
            _final_res.to_csv(f'./area_feature/area_{i}_{flag}.csv')  # output to csv


