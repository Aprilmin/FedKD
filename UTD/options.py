import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--framework', type=str, default='distributed',choices=['centralized','distributed'])
    parser.add_argument('--scenarios', type=str, default='One-to-One',choices=['Heterogeneous', 'Homogeneous','One-to-One'])


    parser.add_argument('--NonIIDscheme', type=str, default='dirichlet',help='dirichlet partition/ pathological partition')
    parser.add_argument('--Naplha',type=float,default=0.1, help='heterogeneity degree of dirichlet/ Each client is randomly assigned limited classes from all classes')


    parser.add_argument('--participate_data_ratio', type=float, default=0.5)
    parser.add_argument('--data_type', type=str, default='full',choices=['full','nointersection','incremental'])
    parser.add_argument('--iid', type=float, default=0.1,help='1:iid 、 0:pathological、 0.x: dirichlet')
    parser.add_argument('--ModalityMixFrac', nargs='+', type=float, default=[0, 0.6, 0.4], help='[L/G/I,LG/GI/LI,LGI]')
    parser.add_argument('--modal_iid', type=int, default=1, help='0: client0 has his and client1 has his+pic; 1 client0 and client1 have his+pic')
    parser.add_argument('--run_type', type=str, default='train', choices=['ablation', 'train'])

    parser.add_argument('--classNum', type=int, default=27, help='two class or five class or three class')

    parser.add_argument('--data_json_path', type=str, default=r'D:\PycharmProjects\mycode\multi-scale MUTAN\UTD\UTD_cosmo')
    parser.add_argument('--cloud_model_type',type=str,default='large',choices=['large', 'base'])
    parser.add_argument('--edge_model_large_frac', type=float,default=1)


    parser.add_argument('--large_model_path', type=str,default=r'D:\PycharmProjects\mycode\2025-DoubleCap\UTD\models\clip-vit-large-patch14')
    parser.add_argument('--base_model_path', type=str,default=r'D:\PycharmProjects\mycode\2025-DoubleCap\UTD\models\clip-vit-base-patch32')



    parser.add_argument('--client_num', type=int, default=5, help='number of users')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--peft', type=str, default='perTucker',choices=['lora', 'perTucker','Tlora'])
    parser.add_argument('--cloud_peft', type=str, default='Tlora', choices=['lora', 'Tlora','perTucker'])
    parser.add_argument('--aggregation', type=str, default='doublecap')

    parser.add_argument('--use_module', nargs='+', type=bool, default=[True,True,True], help="[pertucker,d2cloud,d2edge]")


    parser.add_argument('--gpa_hidden_size',type=int,default=128)
    parser.add_argument('--d2edge_epochs', type=int, default=1, help="number of rounds of training")
    parser.add_argument('--seed_num', type=int, default=42)
    parser.add_argument('--result_dir_name', type=str, default='result')
    parser.add_argument('--startC', type=int, default=0, help="number of rounds of training")
    parser.add_argument('--endC', type=int, default=100, help="number of rounds of training")
    parser.add_argument('--cloud_epochs', type=int, default=4, help="number of rounds of training")
    parser.add_argument('--epochs', type=int, default=3, help="number of rounds of training")
    parser.add_argument('--communications', type=int, default=100, help="number of rounds of training")
    parser.add_argument('--edge_bs',type=int, default=64, help="[edge0,edge1,cloud]") #64
    parser.add_argument('--gcs', type=int, default=2, help="gradient_accumulation_steps")
    parser.add_argument('--gpus', type=int, default=1)


    parser.add_argument('--min_lr', type=float, default=1e-8, help="[edge0,edge1]") # 1e-5
    parser.add_argument('--edge_lr', type=float, default= 0.001, help="[edge0,edge1]")
    parser.add_argument('--d2cloud_lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--d2edge_lr', type=float, default=0.005, help='learning rate')


    parser.add_argument('--edge_alpha', type=float, default=0.3, help='learning rate')





    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    argsDict = args.__dict__


    return args,argsDict
