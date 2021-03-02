from pandas.core.indexing import is_label_like
from typing_extensions import final
from config import *
from toolkit import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def cacu_features(df:pd.DataFrame, window:int, step:int,tag:str,eqData:pd.DataFrame,flag:str) ->pd.DataFrame:
    '''
    生成训练数据。对给定区域，以window(单位：秒)为窗口长生成特征
    '''
    if(len(df)==0):
        return None
    df.reset_index(drop=True,inplace=True)
    averageName = tag+'@abs_mean'
    df.rename(columns={averageName:'average'},inplace=True)
    df['average'] = df['average']-df['average'].mean()
    df['diff_1'] = df.groupby('StationID')['average'].shift(1)
    df['diff_1'] = df['average'].values - df['diff_1'].values

    df.loc[:,'Day'] = df['TimeStamp']
    _start_timestamp = string2stamp(Time_Range[flag][0])
    df['Day'] = df['Day'] - _start_timestamp
    df['Day'] = (df['Day']//86400+1).astype(int)
    df.reset_index()
    tmp = pd.DataFrame(sorted(df['Day'].unique()))
    tmp.columns=['Day']
    res_df = pd.DataFrame((tmp['Day']//step+1).unique()).astype(int)
    res_df.columns=['Day']
    res_df['Day'] = res_df['Day']*step
    for feature in ['average', 'diff_1']:
        for tagging in ['max', 'min', 'mean']:
            kk = df.groupby('Day')[feature].agg(tagging)
            kk.rename(f'{feature}_day_{tagging}',inplace=True)
            tmp = pd.merge(tmp, kk, how='left',on='Day')
        #max_mean.min_mean:
        tmp[f'{feature}_day_max_mean'] = tmp[f'{feature}_day_max'].rolling(window=window,center=False).mean()
        tmp[f'{feature}_day_min_mean'] = tmp[f'{feature}_day_min'].rolling(window=window,center=False).mean()
        #mean_max,mean_min:
        tmp[f'{feature}_day_mean_max'] = tmp[f'{feature}_day_mean'].rolling(window=window,center=False).max()
        tmp[f'{feature}_day_mean_min'] = tmp[f'{feature}_day_mean'].rolling(window=window,center=False).min()
        res_df = pd.merge(res_df,tmp[['Day',f'{feature}_day_max_mean',f'{feature}_day_min_mean',f'{feature}_day_mean_max',f'{feature}_day_mean_min']],on='Day',how='left')


        res_df[f'{feature}_mean'] = None
        res_df[f'{feature}_max'] = None
        res_df[f'{feature}_min'] = None
        res_df[f'{feature}_max_min'] = None
        for i,row in res_df.iterrows():
            endDay = row['Day']
            startDay = endDay - window
            data_se = df[(df['Day']>startDay)&(df['Day']<=endDay)][feature]
            res_df[f'{feature}_mean'].iloc[i] = data_se.mean()
            res_df[f'{feature}_max'].iloc[i] = data_se.max()
            res_df[f'{feature}_min'].iloc[i] = data_se.min()
            res_df[f'{feature}_max_min'].iloc[i] = data_se.max() - data_se.min()


        res_df[f'{feature}_lastday_mean'] = None
        res_df[f'{feature}_lastday_max'] = None
        res_df[f'{feature}_lastday_min'] = None
        res_df[f'{feature}_lastday_max_min'] = None
        for i,row in res_df.iterrows():
            endDay = row['Day']
            data_last = df[df['Day']==endDay][feature]
            res_df[f'{feature}_lastday_mean'].iloc[i] = data_last.mean()
            res_df[f'{feature}_lastday_max'].iloc[i] = data_last.max()
            res_df[f'{feature}_lastday_min'].iloc[i] = data_last.min()
            res_df[f'{feature}_lastday_max_min'].iloc[i] = data_last.max() - data_last.min()
    
    for name in res_df.columns.to_list():
        if(name=='Day'):continue
        res_df.rename(columns={name:(name+'_'+tag)},inplace=True)
    res_df.dropna(axis=0,how='any',inplace=True)
    res_df.reset_index(drop=True,inplace=True)
    res_df['label_M'] = None
    res_df['label_long'] = None
    res_df['label_lati'] = None
    #起始时间戳
    zero_stamp = _start_timestamp
    for i,row in res_df.iterrows():
        endDay = row['Day']
        endStamp = zero_stamp + (endDay-1)*86400
        pre_Range_left = endStamp+86400*2   
        pre_Range_right = endStamp+86400*9   #左闭右开区间,注意，数据窗口和预测周之间隔着周日，周日数据无法即时取得
        #使用时间范围内，区域范围内最大震级的地震作为label 
        _eq = eqData[(eqData['Timestamp']<pre_Range_right) & (eqData['Timestamp']>=pre_Range_left)]
        if(len(_eq)==0):
            res_df['label_M'].iloc[i] = 0
            res_df['label_long'].iloc[i] = -1
            res_df['label_lati'].iloc[i] = -1
        else:
            _eq_max = _eq.iloc[_eq['Magnitude'].argmax()]
            res_df['label_M'].iloc[i] = _eq_max['Magnitude']
            res_df['label_long'].iloc[i] = _eq_max['Longitude']
            res_df['label_lati'].iloc[i] = _eq_max['Latitude']
    return res_df

    
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

        local_magn_data = magn_data[magn_data['StationID'].apply(lambda x:x in ID_list)].reset_index(drop=True)
        local_sound_data = sound_data[sound_data['StationID'].apply(lambda x:x in ID_list)].reset_index(drop=True)
        for flag in ['train', 'valid']:
            time_range = Time_Range[flag]
            start_stamp = string2stamp(time_range[0])
            end_stamp = string2stamp(time_range[1])
            _df_magn = local_magn_data[(local_magn_data['TimeStamp']>=start_stamp)&(local_magn_data['TimeStamp']<end_stamp)]
            _df_sound = local_sound_data[(local_sound_data['TimeStamp']>=start_stamp)&(local_sound_data['TimeStamp']<end_stamp)]
            _magn_res = cacu_features(_df_magn,Window,Step,'magn',eqData_area,flag)
            _sound_res = cacu_features(_df_sound,Window,Step,'sound',eqData_area,flag)
            #抛弃重合的label列/ drop surplus column
            _magn_res.drop(['label_M','label_long','label_lati'],axis=1,inplace=True)
            _final_res = pd.merge(_magn_res,_sound_res,on='Day',how='left')
            _final_res.dropna(inplace=True)
            _final_res.to_csv(f'./area_feature/area_{i}_{flag}.csv')


