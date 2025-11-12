import torch
from itertools import combinations
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,hidden_num,feature_out_num):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(hidden_num,feature_out_num,bias=False)

    def forward(self,inputData):

        classifierOut=self.classifier(inputData)
        if len(classifierOut.shape)==1:
            classifierOut=torch.unsqueeze(classifierOut,dim=0)
        # classifierOut=F.softmax(classifierOut,dim=1)
        classifierOut=nn.functional.normalize(classifierOut,dim=1)
        return classifierOut
class skeSubnet(nn.Module):

    def __init__(self, input_size,output_size):
        super().__init__()

        # Extract features, 3D conv layers
        self.features = nn.Sequential(
            nn.Conv3d(input_size, 64, [5,5,2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(64, 64, [5,5,2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(64, 32, [5,5,1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(32, 16, [5,2,1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            )
        self.fc=nn.Sequential(
            nn.Linear(in_features=16*4*27,out_features=output_size),

        )


    def forward(self, x):
        x=x.permute(0,1,3,2)
        x=torch.unsqueeze(x,1)
        x = self.features(x)
        # print(x.shape)
        x=x.view(x.shape[0],-1)

        x=self.fc(x)
        x=nn.functional.normalize(x,dim=1)

        return x
class inertialSubNet(nn.Module):  # 文本的基于LSTM的子网

    def __init__(self, in_size, hidden_size,  num_layers=1, dropout=0.0, bidirectional=False):

        super(inertialSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout,
                           bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):

        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        h=nn.functional.normalize(h,dim=1) if len(h.shape)==2 else torch.unsqueeze(h,0)
        # y_1 = self.linear_1(h)
        return h
class depthSubnet(nn.Module):
    def __init__(self,out_size,input_size=1,dropout=0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_size,out_channels=64,kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.AvgPool2d(3),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),


            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            )
        self.fc=nn.Sequential(
            nn.Linear(in_features=16*5*10,out_features=out_size),

        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        x=nn.functional.normalize(x,dim=1)
        return x


class FeatureEmbedding(nn.Module):
    def __init__(self,hiddenNum):
        super(FeatureEmbedding, self).__init__()
        self.hiddenNum=hiddenNum
        self.featureEmbeddingList=nn.ModuleList([
            depthSubnet(hiddenNum),  # depth 0.001 ac=0.5
            inertialSubNet(120, hiddenNum, dropout=0.5),  # inertial lr=0.01 ac=0.625,
            skeSubnet(input_size=1, output_size=hiddenNum),  # lr=0.01 ac=0.7

        ])

    def forward(self,inputData,subnetFlag):
        '''
        inpurData:(depth,inertial,skeleton)
        subnetFlag:[depth,inertial,skeleton]
        '''
        bs=inputData[0].shape[0]
        ModalityFeatures=[self.featureEmbeddingList[modalityIdx](inputData[modalityIdx]) if subnetFlag[modalityIdx]==1 else torch.zeros(bs,self.hiddenNum).cuda() for modalityIdx in range(len(subnetFlag))]
        return ModalityFeatures
class biMutanFusion(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers):
        super(biMutanFusion, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)

            hv.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.image_transformation_layers = nn.ModuleList(hv)
        #
        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.ques_transformation_layers = nn.ModuleList(hq)

    def forward(self, ques_emb, img_emb):
        # Pdb().set_trace()
        batch_size = img_emb.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)

            x_hq = ques_emb
            x_hq = self.ques_transformation_layers[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))
        #
        x_mm = torch.stack(x_mm, dim=1)
        x_mm=x_mm.sum(1)
        x_mm = x_mm.view(batch_size, self.out_dim)
        x_mm = nn.functional.tanh(x_mm)
        return x_mm

class MSMutanExpert(nn.Module):
    def __init__(self,hiddenNum,rank):
        super(MSMutanExpert, self).__init__()

        self.modalityNum = 3
        self.modalityCombination = [[modalityIdx, modalityIdx] for modalityIdx in range(3)]
        self.modalityCombination.extend([list(c) for c in combinations(range(self.modalityNum), 2)])

        self.featEmbedding = FeatureEmbedding(hiddenNum)
        self.modalityFusion = biMutanFusion(hiddenNum, hiddenNum, rank)




    def forward(self, inputData, subnetFlag):


        bs = inputData[0].shape[0]
        ModalityFeatues = self.featEmbedding(inputData, subnetFlag)
        ModalityFeatues = [torch.unsqueeze(ModalityFeatues[modality], dim=0) if len(ModalityFeatues[modality].shape) == 1 else ModalityFeatues[modality] for modality in range(len(ModalityFeatues))]

        combinationFeature = [self.modalityFusion(ModalityFeatues[modalitySet[0]], ModalityFeatues[modalitySet[1]]) for modalitySet in self.modalityCombination]
        combinationFeature = torch.stack(combinationFeature, dim=1)
        combinationFeature_norm = torch.norm(combinationFeature, dim=2, keepdim=True)
        multiScaleMutanOut = combinationFeature / combinationFeature_norm





        return multiScaleMutanOut,ModalityFeatues
class FeatureFusionModel(nn.Module):
    def __init__(self,hiddenNum,fusion_rank,feature_out_num):
        super(FeatureFusionModel, self).__init__()
        self.feature=MSMutanExpert(hiddenNum,fusion_rank)
        self.classifierLayer=Classifier(hiddenNum,feature_out_num)

    def forward(self,inputData,subnetFlag):


        fused_feature,modality_feature=self.feature(inputData,subnetFlag)
        featureSize = fused_feature.shape
        classifierOut=[self.classifierLayer(torch.squeeze(fused_feature[:,modalityIdx,:])) for modalityIdx in range(featureSize[1])]
        classifierOut= torch.stack(classifierOut, dim=0).mean(0)
        return classifierOut,modality_feature,fused_feature