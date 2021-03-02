#路径参数

Data_Folder_Path = './data/'                      #原始数据文件夹
Eq_list_path = Data_Folder_Path+'eqlst.csv'               #地震目录文件
Station_Info_Path = Data_Folder_Path+'StationInfo.csv'      #台站信息文件
Usable_Station_Path = Data_Folder_Path + 'UsableStation.bin'    #可用台站集

Used_features = {'magn':['StationID', 'TimeStamp', 'magn@abs_mean'],
                    'sound':['StationID', 'TimeStamp', 'sound@abs_mean']}#选取电磁、地声的绝对值均值特征

#合并数据文件
Merged_Data_Path= {'magn':Data_Folder_Path + 'magn_data.pkl',
                    'sound':Data_Folder_Path + 'sound_data.pkl'}

#训练集与验证集生成
Time_Range = {'train':['20161001','20200331'],              #训练集时间段
                'valid':['20200401','20201231']}                #验证集时间段                        
Window = 7                                                      #窗长，单位 天
Step = 7                                                        #步长，单位 天