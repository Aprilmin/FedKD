import os
import json
import numpy as np
from tqdm import tqdm
class data_processor():
    def __init__(self,raw_processed_data_dir):
        if not os.path.exists(r'./data/split_data'):
            os.makedirs(r'./data/split_data')
        with open(os.path.join(raw_processed_data_dir,'train_data.json'),'r') as f:
            self.train_data=json.load(f)
        with open(os.path.join(raw_processed_data_dir,'test_data_edge.json'),'r') as f:
            self.test_data_edge=json.load(f)
        self.weather_words=['clear', 'foggy', 'cloudy', 'rainy', 'overcast']

    def distributed_data_processor(self,root_path,participate_data_ratio=0.5,client_num=2,iid=0,data_type='nointersection',communication=20):

        if not os.path.exists(root_path):
            print('################################ generating distributed data #################################')
            saved_cloud_ratio=participate_data_ratio*1/(client_num+1)   # cloud:[：cloud_ratio] edges:[cloud_ratio:participate_data_ratio]
            saved_cloud_train_data=[]
            saved_cloud_test_data=[]
            saved_edge_train_data={f'Client{client_idx}':[] for client_idx in range(client_num)}
            saved_edge_test_data={f'Client{client_idx}':[] for client_idx in range(client_num)}


            for weather in self.train_data.keys():
                all_scenes_token=list(self.train_data[weather].keys())
                saved_cloud_token=all_scenes_token[:int(len(all_scenes_token)*saved_cloud_ratio)]
                saved_edge_token=all_scenes_token[int(len(all_scenes_token)*saved_cloud_ratio):int(len(all_scenes_token)*participate_data_ratio)]

                # cloud
                for scene_token in tqdm(saved_cloud_token):
                    # cloud test
                    if self.test_data_edge[weather][scene_token] is not None:
                        for sample in self.test_data_edge[weather][scene_token]:
                            sample['weather']={
                                'description':weather,
                                'label': self.weather_words.index(weather) + 1
                            }
                            saved_cloud_test_data.append(sample)
                    # cloud train
                    for sample in self.train_data[weather][scene_token]:
                        sample['weather'] = {
                            'description': weather,
                            'label': self.weather_words.index(weather) + 1
                        }
                        saved_cloud_train_data.append(sample)



                proportions = np.random.dirichlet(np.repeat(iid, client_num))
                client_token_idx = np.split(np.arange(len(saved_edge_token)), (np.cumsum(proportions) * len(saved_edge_token)).astype(int)[:-1])
                for client_idx in range(client_num):
                    for scene_idx in client_token_idx[client_idx]:
                        this_scene_token=saved_edge_token[scene_idx]
                        if self.test_data_edge[weather][this_scene_token] is not None:
                            for sample in self.test_data_edge[weather][this_scene_token]:
                                sample['weather']={
                                    'description':weather,
                                    'label':self.weather_words.index(weather) + 1
                                }
                                saved_edge_test_data[f'Client{client_idx}'].append(sample)
                        for sample in self.train_data[weather][this_scene_token]:
                            sample['weather'] = {
                                'description': weather,
                                'label': self.weather_words.index(weather) + 1
                            }
                            saved_edge_train_data[f'Client{client_idx}'].append(sample)

            if data_type.lower().strip()!='full':
                samples_per_period = int(len(saved_cloud_train_data) / communication)
                edge_train_data_by_time = []
                current_sample = None
                for c in range(communication):
                    sample_for_this_round = saved_cloud_train_data[int(c * samples_per_period):int((c + 1) * samples_per_period)] if c != communication - 1 else saved_cloud_train_data[int((c) * samples_per_period):]
                    if data_type.lower().strip() == 'incremental':
                        if current_sample is None:
                            current_sample = sample_for_this_round
                        else:
                            current_sample.extend(sample_for_this_round)
                        edge_train_data_by_time.append(list(current_sample))
                    elif data_type.lower().strip() == 'nointersection':
                        edge_train_data_by_time.append(sample_for_this_round)
                saved_cloud_train_data = edge_train_data_by_time

                for client_idx in range(client_num):
                    samples_per_period = int(len(saved_edge_train_data[f'Client{client_idx}']) / communication)
                    edge_train_data_by_time=[]
                    current_sample=None
                    for c in range(communication):
                        sample_for_this_round=saved_edge_train_data[f'Client{client_idx}'][int(c*samples_per_period):int((c+1)*samples_per_period)] if c!=communication-1 else saved_edge_train_data[f'Client{client_idx}'][int((c)*samples_per_period):]
                        if data_type.lower().strip()=='incremental':
                            if current_sample is None:
                                current_sample=sample_for_this_round
                            else:
                                current_sample.extend(sample_for_this_round)
                            edge_train_data_by_time.append(list(current_sample))
                        elif data_type.lower().strip()=='nointersection':
                            edge_train_data_by_time.append(sample_for_this_round)
                    saved_edge_train_data[f'Client{client_idx}']=edge_train_data_by_time
            if data_type.lower().strip()=='full':
                file_path=os.path.join(root_path,'Cloud')
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                with open(os.path.join(file_path,'train.json'),'w') as f:
                    json.dump(saved_cloud_train_data,f,indent=4)
                with open(os.path.join(file_path,'test.json'),'w') as f:
                    json.dump(saved_cloud_test_data,f,indent=4)


                for client_idx in range(client_num):
                    client_name=f'Client{client_idx}'
                    file_path=os.path.join(root_path,client_name)
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    with open(os.path.join(file_path, 'train.json'), 'w') as f:
                        json.dump(saved_edge_train_data[client_name], f, indent=4)
                    with open(os.path.join(file_path, 'test.json'), 'w') as f:
                        json.dump(saved_edge_test_data[client_name], f, indent=4)
            else:
                file_path=os.path.join(root_path,'Cloud')
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                for c in range(communication):
                    with open(os.path.join(file_path,f'{c}_train.json'),'w') as f:
                        json.dump(saved_cloud_train_data[c],f,indent=4)
                with open(os.path.join(file_path,'test.json'),'w') as f:
                    json.dump(saved_cloud_test_data,f,indent=4)

                for client_idx in range(client_num):

                    client_name=f'Client{client_idx}'
                    print(f'******************************generated {client_name} data*********************')
                    file_path=os.path.join(root_path,client_name)
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    for c in range(communication):
                        with open(os.path.join(file_path, f'{c}_train.json'), 'w') as f:
                            json.dump(saved_edge_train_data[client_name][c], f, indent=4)
                    with open(os.path.join(file_path, 'test.json'), 'w') as f:
                        json.dump(saved_edge_test_data[client_name], f, indent=4)



