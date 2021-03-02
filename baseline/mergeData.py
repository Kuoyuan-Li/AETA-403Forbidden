import os
import pandas as pd
import re
from tqdm import tqdm  # visualize process bar
from config import *
from toolkit import *

if __name__ == "__main__":
    fileName_list = os.listdir(Data_Folder_Path)  # './data/', data file
    stationInfo_list = pd.read_csv(Station_Info_Path)  # ./data/StationInfo.csv, read the station info file

    # 筛选可用台站：选取同时有地声、电磁数据且在eqlst.csv内标记为未来可用的台站
    _continueable_stations = stationInfo_list[stationInfo_list['MagnUpdate']&stationInfo_list['SoundUpdate']]['StationID'].unique()
    _continueable_stations = set(_continueable_stations)  # Store the continueable stations IDs in a set
    re_magn = re.compile(r'(\d+)_magn.csv')
    re_sound = re.compile(r'(\d+)_sound.csv')
    _set_magn = set()
    _set_sound = set()
    for filename in fileName_list:
        _magn_match = re_magn.findall(filename)  # find all files with _magn suffix in data directory
        _sound_match = re_sound.findall(filename)  # find all files with _sound suffix in data directory
        if(_magn_match):
            _set_magn.add(int(_magn_match[0]))  # add all _magn file Station IDs (column 0) to the set
            continue
        if(_sound_match):
            _set_sound.add(int(_sound_match[0]))  # add all _sound file Station IDs (column 0) to the set
            continue
    usable_stations = _continueable_stations&_set_magn&_set_sound  # set of usable station IDs
    dump_object(Usable_Station_Path, usable_stations)  # output usable station IDs to data/UsableStation.bin

    print('合并数据:')
    for type in ('magn', 'sound'):
        res = []
        for _id in tqdm(usable_stations, desc=f'{type}:'):  # for each station ID in usable station IDs
            _df = pd.read_csv(Data_Folder_Path+str(_id)+f'_{type}.csv')[Used_features[type]]  # read csv file: ./data/<station_id>_<type>.csv, find the needed features data (defined in config) of each station, store in a dataframe'
            res.append(_df)  # add the dataframe of each station to res list
        final_df = pd.concat(res)  # concatenate all station DF in to one dataframe
        final_df.to_pickle(Merged_Data_Path[type])  # store the dataframe in pickle files: magn_data.pkl and sound_data.pkl, for easier reading next time
        del(final_df)
    
    