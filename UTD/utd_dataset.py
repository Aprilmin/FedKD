from torch.utils.data import Dataset
import torch
class SingleModalDataset(Dataset):

    def __init__(self, data,trainFlag,prompt_template):

        self.data=data
        self.trainFlag=trainFlag
        self.prompt_template=prompt_template



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'labels':self.data[idx]['labels'],
            'depth':self.data[idx]['depth'],
            'inertial':self.data[idx]['inertial'],
            'skeleton':self.data[idx]['skeleton'],
            'color':self.data[idx]['color'],
            'selected_idx':idx
        }

    def custom_collate_fn(self, features):

        labels = [f['labels'] for f in features]
        depth = [torch.tensor(f['depth']) for f in features]
        inertial = [torch.tensor(f['inertial']) for f in features]
        skeleton = [torch.tensor(f['skeleton']) for f in features]
        color = [torch.tensor(f['color']) for f in features]
        selected_idx=[f['selected_idx'] for f in features]


        return torch.stack(color,dim=0), torch.stack(depth,dim=0), torch.stack(inertial,dim=0), torch.stack(skeleton,dim=0), torch.tensor(labels).long(),self.prompt_template,self.trainFlag,selected_idx


class MultiFrameDataset(Dataset):

    def __init__(self, input_file,trainFlag=None,prompt_template=None):

        node_data=torch.load(input_file)
        self.data=node_data['data']
        self.trainFlag=node_data['trainFlag'] if trainFlag is None else trainFlag
        self.prompt_template=node_data['prompt_template'] if prompt_template is None else prompt_template


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'labels':self.data[idx]['labels'],
            'depth':self.data[idx]['depth'],
            'inertial':self.data[idx]['inertial'],
            'skeleton':self.data[idx]['skeleton'],
            'color':self.data[idx]['color'],
            'selected_idx':idx
        }

    def custom_collate_fn(self, features):

        labels = [f['labels'] for f in features]
        depth = [torch.tensor(f['depth']) for f in features]
        inertial = [torch.tensor(f['inertial']) for f in features]
        skeleton = [torch.tensor(f['skeleton']) for f in features]
        color = [torch.tensor(f['color']) for f in features]
        selected_idx=[f['selected_idx'] for f in features]


        return torch.stack(color,dim=0), torch.stack(depth,dim=0), torch.stack(inertial,dim=0), torch.stack(skeleton,dim=0), torch.tensor(labels).long(),self.prompt_template,self.trainFlag,selected_idx

class Edges2CloudMultiFrameDataset(Dataset):

    def __init__(self, cloud_data,teacher_data,prompt_template,trainFlag):
        self.data=cloud_data
        self.prompt_template = prompt_template

        self.teachers_data=[]
        self.teachers_class=[]
        for teacher_idx in range(len(teacher_data)):
            self.teachers_data.append(teacher_data[teacher_idx]['teacher_data'])
            self.teachers_class.append(teacher_data[teacher_idx]['teacher_class'])
        self.client_num=len(teacher_data)
        self.trainFlag=trainFlag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'labels': self.data[idx]['labels'],
            'depth': self.data[idx]['depth'],
            'inertial': self.data[idx]['inertial'],
            'skeleton': self.data[idx]['skeleton'],
            'color': self.data[idx]['color'],
            'selected_idx': idx,

            'teacher_personal_feat':[self.teachers_data[client_idx][idx]['last_peft_output']['personal'] for client_idx in range(self.client_num)],
            'teacher_fusion_feat':[self.teachers_data[client_idx][idx]['fusion_feat'] for client_idx in range(self.client_num)],
            'teacher_text_feat':[self.teachers_data[client_idx][idx]['text_feat'] for client_idx in range(self.client_num)],
            'teachers_logit': [self.teachers_data[client_idx][idx]['local_output'] for client_idx in range(self.client_num)],
            'teachers_class': torch.tensor(self.teachers_class)
        }

    def custom_collate_fn(self, features):
        labels = torch.tensor([f['labels'] for f in features]).long()
        depth = torch.stack([torch.tensor(f['depth']) for f in features],dim=0)
        inertial = torch.stack([torch.tensor(f['inertial']) for f in features],dim=0)
        skeleton = torch.stack([torch.tensor(f['skeleton']) for f in features],dim=0)
        color = torch.stack([torch.tensor(f['color']) for f in features],dim=0)
        selected_idx=[f['selected_idx'] for f in features]


        teacher_personal_feat=[f['teacher_personal_feat'] for f in features]
        teacher_fusion_feat = [f['teacher_fusion_feat'] for f in features]
        teacher_text_feat = [f['teacher_text_feat'] for f in features]
        teachers_logit = torch.stack([torch.stack(f['teachers_logit'],dim=0) for f in features], dim=0)



        return color, depth,inertial ,skeleton , labels,self.prompt_template,self.trainFlag,selected_idx,teacher_personal_feat,teacher_fusion_feat,teacher_text_feat,teachers_logit,self.teachers_class,self.client_num


class Cloud2EdgesMultiFrameDataset(Dataset):

    def __init__(self, data, trainFlag, prompt_template, teachers_data):


        self.data=data
        self.trainFlag=trainFlag
        self.prompt_template=prompt_template
        self.teachers_data = teachers_data
        self.check_effective_data()


    def check_effective_data(self):
        sample_num = self.__len__()
        effective_data = []
        effective_teachers_data = []
        for idx in range(sample_num):
            if self.teachers_data[idx]['local_output'] is not None:
                effective_data.append(self.data[idx])
                effective_teachers_data.append(self.teachers_data[idx])
        self.data = effective_data
        self.teachers_data = effective_teachers_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'labels': self.data[idx]['labels'],
            'depth': self.data[idx]['depth'],
            'inertial': self.data[idx]['inertial'],
            'skeleton': self.data[idx]['skeleton'],
            'color': self.data[idx]['color'],
            'selected_idx': idx,

            'teachers_logit': self.teachers_data[idx]['local_output']
        }

    def custom_collate_fn(self, features):
        labels = torch.tensor([f['labels'] for f in features]).long()
        depth = torch.stack([torch.tensor(f['depth']) for f in features],dim=0)
        inertial = torch.stack([torch.tensor(f['inertial']) for f in features],dim=0)
        skeleton = torch.stack([torch.tensor(f['skeleton']) for f in features],dim=0)
        color = torch.stack([torch.tensor(f['color']) for f in features],dim=0)
        selected_idx=[f['selected_idx'] for f in features]


        teachers_logit = torch.stack([f['teachers_logit'] for f in features], dim=0)


        return color, depth,inertial ,skeleton , labels,self.prompt_template,self.trainFlag,selected_idx,teachers_logit



