import os
import json
import numpy as np
import shutil
from itertools import combinations
import pandas as pd
import _pickle as cPickle
import math
import torch
class data_processor():
    def __init__(self,dirPath=r'D:\PycharmProjects\mycode\multi-scale MUTAN\UTD\UTD_cosmo',args=None,subnetFlagMap=['D','I','S']):
        self.subnetFlagMap=subnetFlagMap
        self.GlobalTestSubnetCombinations = self.getGlobalCombinations(args)
        self.ServerTestResult, self.ClientDataDistribution = self.getResultDict(args)
        self.ClientTrainingSubnetFlag = self.getLocalTrainingSubnetFlag(args)
        self.dirPath=dirPath
        self.templates, self.action_label_to_text, self.text_prompt_list=self.getTextPrompt()


    def getTextPrompt(self):
        templates = [
            "a action of {}",
            "a person {}",
            "a demonstration of the {} action",
            "someone {}",
            "the gesture of {}"
        ]
        action_label_to_text=[
            'Swipe left','Swipe right','Wave','Clap','Throw',
            'Arm cross','Basketball shoot','Draw X','Draw circle(clockwise)','Draw circle(counter clockwise)',
            'Draw triangle','Bowling','Boxing','Baseball swing','Tennis swing',
            'Arm curl','Tennis serve','Push','Knock','Catch',
            'Pickup and throw','Jog','Walk','Sit to stand','Stand to sit',
            'Lunge','Squat'
        ]
        text_prompt_list=[]
        for templateIdx in range(len(templates)):
            template=templates[templateIdx]
            text_prompt_list.append([
                template.format(action_label_to_text[actionIdx]) for actionIdx in range(len(action_label_to_text))
            ])
        return templates,action_label_to_text,text_prompt_list


    def getResultDict(self,args):
        ServerTestResult={
            key:{
                'accuracy':[],
                'precisionAVG':[],
                'precisionWei': [],
                'recallAVG':[],
                'f1-score': [],
                'loss':[]
            } for key in ['D','I','S','D+S','D+I','I+S','D+I+S','M1','M2','M3','ModalityAvg']
        }
        ServerTestResult['OtherMetrics']={
            'LocalTrainingTime':[],
            'ServerTrainingTime':[],
            'AverageTestTime':[],
            'SendParametersSize':[],
            'LocalTestAverageAcc': [],
            'LocalTestAverageMap': [],
            'LocalTestAverageMapWei': []
        }
        columnsName=['client_{}'.format(userIdx) for userIdx in range(args.client_num)]
        columnsName.append('cloud')
        columnsName.append('supportExamples')
        indexName=list(range(args.classNum))
        indexName.append('Modality')

        ClientDataDistribution=pd.DataFrame(columns=columnsName,index=indexName).fillna(0)
        return ServerTestResult,ClientDataDistribution
    def getLocalTrainingSubnetFlag(self,args):
        ClientIdxSet=np.arange(args.client_num)
        split_indices = (len(ClientIdxSet) * np.array(args.ModalityMixFrac)).astype(int)
        ClientSplit = np.split(ClientIdxSet, np.cumsum(split_indices)[:-1])
        ClientTrainingSubnetFlag=[np.zeros(len(args.ModalityMixFrac)) for _ in range(args.client_num)]
        for ModalityNum in range(len(args.ModalityMixFrac)):
            if len(ClientSplit[ModalityNum])!=0:
                for clientIdx in ClientSplit[ModalityNum]:
                    ClientTrainingSubnetFlag[clientIdx]=self.GlobalTestSubnetCombinations[ModalityNum][int(np.random.choice(range(len(self.GlobalTestSubnetCombinations[ModalityNum])),1)[0])]

                    self.ClientDataDistribution.loc['Modality', 'client_{}'.format(clientIdx)] = '{}/{}'.format(''.join([self.subnetFlagMap[idx] for idx in np.flatnonzero(np.array(ClientTrainingSubnetFlag[clientIdx]) == 1)]),'-'.join([str(int(ClientTrainingSubnetFlag[clientIdx][i])) for i in range(len(ClientTrainingSubnetFlag[clientIdx]))]))

        unq,unq_count=np.unique(np.array(self.ClientDataDistribution.loc['Modality'][:-1].to_list()),return_counts=True)

        return ClientTrainingSubnetFlag
    def getGlobalCombinations(self,args):
        subnetFlagCombinations = [[] for _ in range(len(args.ModalityMixFrac))]
        for sampleNum in range(len(args.ModalityMixFrac)):
            c = combinations(range(len(args.ModalityMixFrac)), sampleNum + 1)
            for sampleIndex in c:
                tmp = np.zeros(len(args.ModalityMixFrac))
                tmp[np.array(sampleIndex)] = 1
                subnetFlagCombinations[sampleNum].append(tmp)
        return subnetFlagCombinations

    def random_cloud_data_processor(self,root_path,ref_path):
        all_nodes_files=os.listdir(ref_path)
        client_list=list(filter(lambda x:'client' in x.lower(),all_nodes_files))

        for client_file in client_list:
            src_path=os.path.join(ref_path,client_file)
            dst_path=os.path.join(root_path,client_file)
            shutil.move(src_path,dst_path)

        train_data=torch.load(os.path.join(ref_path,'Cloud','train.pt'))
        cloud_data=torch.load(os.path.join(ref_path,'Cloud','test.pt'))['data']+train_data['data']

        cloud_indices=np.arange(len(cloud_data))

        all_subnet_list, unq_count = np.unique(self.ClientTrainingSubnetFlag, axis=0, return_counts=True)


        # trainFlag = self.ClientTrainingSubnetFlag[client_num]
        node_name = 'Cloud'

        np.random.shuffle(cloud_indices)
        split_modality_indices = np.hsplit(
            cloud_indices[:int(len(cloud_indices) // len(all_subnet_list) * len(all_subnet_list))],len(all_subnet_list))
        cloud_train_data = {}
        for trainFlag_idx in range(len(all_subnet_list)):
            trainFlag = all_subnet_list[trainFlag_idx]
            modality_train_data = []
            modality_train_indice = split_modality_indices[trainFlag_idx]
            for sampleIdx in modality_train_indice:
                modality_train_data.append({
                    'labels': cloud_data[sampleIdx]['labels'],
                    'depth': cloud_data[sampleIdx]['depth'],
                    'inertial': cloud_data[sampleIdx]['inertial'],
                    'skeleton': cloud_data[sampleIdx]['skeleton'],
                    'color': cloud_data[sampleIdx]['color'],
                    'label2text': cloud_data[sampleIdx]['label2text'],
                    'sampleIdx': cloud_data[sampleIdx]['sampleIdx']
                })
            trainFlag_str = '-'.join([str(int(trainFlag[i])) for i in range(len(trainFlag))])
            cloud_train_data[trainFlag_str] = modality_train_data
        if not os.path.exists(os.path.join(root_path, node_name)):
            os.makedirs(os.path.join(root_path, node_name))
        torch.save({
            'data': cloud_train_data,
            'prompt_template': train_data['prompt_template'],
        }, os.path.join(os.path.join(root_path, node_name), 'train.pt'))






    def distributed_data_processor(self,root_path,client_num,iid):
        classNum=27
        test_ratio = 0.2
        if not os.path.exists(root_path):
            print('################################ generating distributed data #################################')

            trainPath = os.path.join(self.dirPath, 'trainUTD.pkl')
            testPath = os.path.join(self.dirPath, 'testUTD.pkl')
            validPath = os.path.join(self.dirPath, 'validUTD.pkl')
            allData = []
            for path in [trainPath, testPath, validPath]:
                f = open(path, 'rb+')
                data = cPickle.load(f)
                allData.append(data)
            color = np.concatenate([allData[i].color for i in range(len(allData))], axis=0)
            depth = np.concatenate([allData[i].depth for i in range(len(allData))], axis=0)
            inertial = np.concatenate([allData[i].inertial for i in range(len(allData))], axis=0)
            skeleton = np.concatenate([allData[i].skeleton for i in range(len(allData))], axis=0)
            labels = np.concatenate([allData[i].labels for i in range(len(allData))], axis=0)

            multiplyNum = math.ceil((client_num+1) / 5)
            indices = np.repeat(np.arange(len(labels)), multiplyNum)
            color = color[indices]
            depth = depth[indices]
            inertial = inertial[indices]
            skeleton = skeleton[indices]
            labels = labels[indices]






            all_indices = np.arange(len(labels))
            np.random.shuffle(all_indices)

            cloud_indices = all_indices[:int(len(all_indices) / (client_num + 1))]
            prompt_idx = np.random.choice(len(self.text_prompt_list), 1)[0]
            self.ClientDataDistribution.loc['prompt_template', 'cloud'] = self.templates[prompt_idx]
            unq, unq_count = np.unique(labels[cloud_indices], return_counts=True)
            client_class_state = np.zeros(classNum)
            client_class_state[unq] = unq_count
            self.ClientDataDistribution.loc[np.arange(classNum), 'cloud'] = client_class_state
            all_subnet_list, unq_count =np.unique(self.ClientTrainingSubnetFlag,axis=0, return_counts=True)


            node_name = 'Cloud'

            np.random.shuffle(cloud_indices)
            split_modality_indices = np.hsplit(cloud_indices[:int(len(cloud_indices)//len(all_subnet_list)*len(all_subnet_list))], len(all_subnet_list))
            cloud_train_data={}
            for trainFlag_idx in range(len(all_subnet_list)):
                trainFlag=all_subnet_list[trainFlag_idx]
                modality_train_data=[]
                modality_train_indice=split_modality_indices[trainFlag_idx]
                for sampleIdx in modality_train_indice:
                    modality_train_data.append({
                        'labels': labels[sampleIdx],
                        'depth': depth[sampleIdx],
                        'inertial': inertial[sampleIdx],
                        'skeleton': skeleton[sampleIdx],
                        'color': color[sampleIdx],
                        'label2text': self.text_prompt_list[prompt_idx][labels[sampleIdx]],
                        'sampleIdx': sampleIdx
                    })
                trainFlag_str='-'.join([str(int(trainFlag[i])) for i in range(len(trainFlag))])
                cloud_train_data[trainFlag_str]=modality_train_data
            if not os.path.exists(os.path.join(root_path, node_name)):
                os.makedirs(os.path.join(root_path, node_name))
            torch.save({
                'data': cloud_train_data,
                'prompt_template': self.text_prompt_list[prompt_idx],
            }, os.path.join(os.path.join(root_path,node_name), 'train.pt'))





            indices = all_indices[int(len(all_indices) / (client_num + 1)):]
            color = color[indices]
            depth = depth[indices]
            inertial = inertial[indices]
            skeleton = skeleton[indices]
            labels = labels[indices]


            clientTrainIdxSet= [[] for _ in range(client_num)]
            clientTestIdxSet = [[] for _ in range(client_num)]


            for classIdx in range(classNum):
                indices=np.flatnonzero(labels==classIdx)
                indices = indices if len(indices) % (client_num) == 0 else np.hstack([indices,np.random.choice(indices,int(client_num - len(indices) % (client_num)),replace=False)])
                proportions = np.random.dirichlet(np.repeat(iid, client_num))
                clientAllIdx = np.split(indices, (np.cumsum(proportions) * len(indices)).astype(int)[:-1])

                for clientIdx in range(client_num):
                    clientTestIdx = clientAllIdx[clientIdx][:int(test_ratio * len(clientAllIdx[clientIdx]))]
                    clientTrainIdx = clientAllIdx[clientIdx][int(test_ratio * len(clientAllIdx[clientIdx])):]

                    self.ClientDataDistribution.loc[classIdx,'client_{}'.format(clientIdx)]=len(clientTrainIdx)
                    clientTrainIdxSet[clientIdx].extend(clientTrainIdx)
                    clientTestIdxSet[clientIdx].extend(clientTestIdx)
                self.ClientDataDistribution.loc[classIdx,'supportExamples']=len(indices)
            for clientIdx in range(client_num):
                node_train_data, node_test_data = [], []
                prompt_idx=np.random.choice(len(self.text_prompt_list),1)[0]

                self.ClientDataDistribution.loc['prompt_template', 'client_{}'.format(clientIdx)] = self.templates[prompt_idx]
                trainFlag = self.ClientTrainingSubnetFlag[clientIdx]
                node_name = f'Client{clientIdx}'
                for sampleIdx in clientTrainIdxSet[clientIdx]:
                    node_train_data.append({
                        'labels': labels[sampleIdx],
                        'depth': depth[sampleIdx],
                        'inertial': inertial[sampleIdx],
                        'skeleton': skeleton[sampleIdx],
                        'color': color[sampleIdx],
                        'label2text':self.text_prompt_list[prompt_idx][labels[sampleIdx]],
                        'sampleIdx':sampleIdx
                    })
                for sampleIdx in clientTestIdxSet[clientIdx]:
                    node_test_data.append({
                        'labels': labels[sampleIdx],
                        'depth': depth[sampleIdx],
                        'inertial': inertial[sampleIdx],
                        'skeleton': skeleton[sampleIdx],
                        'color': color[sampleIdx],
                        'label2text':self.text_prompt_list[prompt_idx][labels[sampleIdx]],
                        'sampleIdx':sampleIdx
                    })
                save_generated_data(
                    generated_train_data=node_train_data,
                    generated_test_data=node_test_data,
                    trainFlag=trainFlag,
                    prompt_template=self.text_prompt_list[prompt_idx],
                    file_path=os.path.join(root_path,node_name)
                )



def save_generated_data(generated_train_data,generated_test_data,trainFlag,prompt_template,file_path):

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    torch.save({
        'data':generated_train_data,
        'trainFlag':trainFlag,
        'prompt_template':prompt_template
    },os.path.join(file_path,'train.pt'))

    torch.save({
        'data':generated_test_data,
        'trainFlag':trainFlag,
        'prompt_template':prompt_template
    },os.path.join(file_path,'test.pt'))


