import os
import pandas as pd
import numpy as np
from jaad_data import JAAD
global imdb
jaad_path = "./"
imdb = JAAD(data_path=jaad_path)
# imdb.extract_and_save_images()
#imdb.get_data_stats()
#imdb.generate_database()

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def createVideoFilenameDataFrame(videoname_path):
    video_namelist = pd.read_csv(videoname_path,header = None)
    video_namelist.columns = ['vdo_ids']
    return video_namelist


def CreateGroundtruthsFiles(video_namelist_df):
    global imdb
    dict_cnt_label = {}
    createFolder('./groundtruths/')
    ### extract groundtruth per video
    for vdo_id in video_namelist_df['vdo_ids']:
        # vdo_id = 'video_0031'
        ## create path ./groundtruths/ to match with YOWO
        vdo_GT_path = ('./groundtruths/'+vdo_id+"/")
        
        createFolder(vdo_GT_path)
        video_df = pd.DataFrame(columns=['frames', 'bbox','occlusion', 'ped_id','no_behavior', 'cross', 'reaction', 'hand_gesture', 'look','action', 'nod','bicycle_motorcycle', 'umbrella', 'phone', 'baby', 'object','stroller_cart'])
        # print(vdo_id)
        ## annotations Handling section
        df_annot = pd.DataFrame(imdb._get_annotations(vdo_id).get('ped_annotations'))
        df_appr = pd.DataFrame(imdb._get_ped_appearance(vdo_id))
        # print(sorted([df_annot.columns]))
        # print(sorted([df_appr.columns]),"\n")
        # break

        ## create frames length dictionary for check the ped_id later for df_appr
        frames_length_dict = {}
    
        for col in sorted(df_appr.columns):
            # print(col," : ",len(df_appr[col][0]))
            if len(df_appr[col][0]) in frames_length_dict:
                # print("has value")
                frames_length_dict.pop(len(df_appr[col][0]))
            elif col[-1] != 'b':
                frames_length_dict[len(df_appr[col][0])] = col
        # print(frames_length_dict)


        # del df_appr
        appr = imdb._get_ped_appearance(vdo_id)
        # print(frames_length_dict)
        for col in sorted(df_annot.columns):
            df_Transpose = pd.DataFrame(df_annot[col]).T
            if df_Transpose.behavior[0] != {}:
                # print(col,' : ',len(df_Transpose['frames'][0]))
                # print(col,' : ',len(df_Transpose['frames'][0]),(df_Transpose['frames'][0]),"\n")
                # print(df_Transpose['frames'][0])
                # print('Behavior is not empty')
                # print(len(df_Transpose['frames'][0]))
                df_new_new = pd.DataFrame([df_Transpose['frames'][0],df_Transpose['bbox'][0],df_Transpose['occlusion'][0]]).T
                df_new_new.columns = ['frames','bbox','occlusion']
                df_new_new['ped_id'] = [df_Transpose['old_id'][0]]*len(df_new_new)
                df_new_new['no_behavior'] = 0
                df_new_new = pd.concat([df_new_new,pd.DataFrame(df_Transpose.behavior[0])],axis = 1)

                # break
                if col[-1] == 'b' :
                    appr_id = col
                    for appr_col in [ 'bicycle_motorcycle']:
                            df_new_new[appr_col] = appr.get(appr_id).get(appr_col)
                else : 
                    if len(df_Transpose['frames'][0]) in frames_length_dict:
                        appr_id = frames_length_dict.pop(len(df_Transpose['frames'][0]))
                        for appr_col in [ 'bicycle_motorcycle']:
                            df_new_new[appr_col] = appr.get(appr_id).get(appr_col)

                video_df = pd.concat([video_df,df_new_new],axis = 0,ignore_index = True)
            
            ### pedes type == ped no action
            if df_Transpose.behavior[0] == {}:
                # print(col,' : ',len(df_Transpose['frames'][0]))
                # print(col,' : ',len(df_Transpose['frames'][0]),(df_Transpose['frames'][0]),"\n")
                # print(df_Transpose['frames'][0])
                # print("detected people with no action")
                # print()
                df_new_new = pd.DataFrame([df_Transpose['frames'][0],df_Transpose['bbox'][0],df_Transpose['occlusion'][0]]).T
                df_new_new.columns = ['frames','bbox','occlusion']
                df_new_new['ped_id'] = [df_Transpose['old_id'][0]]*len(df_new_new)
                df_new_new['no_behavior'] = "no_action"
                df_new_new = pd.concat([df_new_new,pd.DataFrame(columns = ['cross', 'reaction', 'hand_gesture', 'look','action', 'nod'])],axis = 1)
                # break
                if col[-1] == 'b' :
                    appr_id = col
                    for appr_col in [ 'bicycle_motorcycle']:
                            df_new_new[appr_col] = appr.get(appr_id).get(appr_col)
                else : 
                    if len(df_Transpose['frames'][0]) in frames_length_dict:
                        appr_id = frames_length_dict.pop(len(df_Transpose['frames'][0]))
                        for appr_col in ['bicycle_motorcycle']: # 'bicycle_motorcycle', 'umbrella', 'phone', 'baby', 'object','stroller_cart'
                            df_new_new[appr_col] = appr.get(appr_id).get(appr_col)
                
                video_df = pd.concat([video_df,df_new_new],axis = 0,ignore_index = True)
        
        video_df = video_df.fillna(0)
        unique_frame = pd.unique(video_df.frames)
        video_df =video_df.sort_values('frames')
        # return video_df
        video_df['label'] = 'no_action'
        for columns_name in ['action','reaction','nod','look','hand_gesture','cross']:
            # print(sum(video_df[columns_name]==1))
            video_df['label'][video_df[columns_name]==1] = columns_name
        video_df.label[video_df['occlusion']==1] += '_occlusion'
        video_df.label[video_df['bicycle_motorcycle']!='0'] += '_bk_mt'
        for lb,cnt in zip(pd.unique(video_df['label']),video_df.groupby('label').count().values[:,0]):
            if dict_cnt_label.get(lb):
                dict_cnt_label[lb] += cnt
            else:
                dict_cnt_label[lb] = cnt
  
        
        for id in unique_frame :
            df = video_df[video_df.frames == id]
            # frame_id = vdo_GT_path+'_'+("00000"+str(id))[-6:]+".txt"
            frame_id =vdo_GT_path+'/'+("00000"+str(id))[-6:]+".txt"
            txt = "{class_name} {left} {top} {right} {bottom} {} "
            # JAAD : two-point coordinates (top-left, bottom-right) [x1, y1, x2, y2]
            # YOWO : <class_name> <left> <top> <right> <bottom>.
            f = open(frame_id, "w")
            txt = "{} {:n} {:n} {:n} {:n}"
            for j in range(len(df)):
                bdbox = df.iloc[j].bbox
                f.write(txt.format((df.iloc[j].cross)+1,bdbox[0]/4,bdbox[1]/4,bdbox[2]/4,bdbox[3]/4))
                f.write("\n")
            f.close()
        print('finish convert groundtruth for', vdo_id)
    print(dict_cnt_label)    
    # break
 


# Train set
train_video_namelist = createVideoFilenameDataFrame('split_ids/all_videos/train.txt')
CreateGroundtruthsFiles(train_video_namelist)
# Validation Set
val_video_namelist = createVideoFilenameDataFrame('split_ids/all_videos/val.txt')
CreateGroundtruthsFiles(test_video_namelist)  
# # Test set
test_video_namelist = createVideoFilenameDataFrame('split_ids/all_videos/test.txt')
CreateGroundtruthsFiles(test_video_namelist)