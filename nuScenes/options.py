import argparse
import distutils.util
import deepspeed
def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--framework', type=str, default='distributed',choices=['centralized','distributed'])
    parser.add_argument('--scenarios', type=str, default='One-to-One',choices=['Heterogeneous', 'Homogeneous','One-to-One'])


    parser.add_argument('--NonIIDscheme', type=str, default='dirichlet',help='dirichlet partition/ pathological partition')
    parser.add_argument('--Naplha',type=float,default=0.1, help='heterogeneity degree of dirichlet/ Each client is randomly assigned limited classes from all classes')
    parser.add_argument('--ModalityMixFrac', nargs='+', type=float, default=[0, 0, 1], help='[L/G/I,LG/GI/LI,LGI]')

    parser.add_argument('--participate_data_ratio', type=float, default=0.5)
    parser.add_argument('--data_type', type=str, default='nointersection',choices=['full','nointersection','incremental'])
    parser.add_argument('--iid', type=float, default=0.1,help='1:iid 、 0:pathological、 0.x: dirichlet')
    parser.add_argument('--modal_iid', type=int, default=1, help='0: client0 has his and client1 has his+pic; 1 client0 and client1 have his+pic')
    parser.add_argument('--run_type', type=str, default='train', choices=['ablation', 'train'])


    parser.add_argument('--raw_data_path', type=str, default=r'./data')
    parser.add_argument('--data_path', type=str,default=r'./data/data_resize0.25')
    parser.add_argument('--data_json_path', type=str, default=r'./data/20250426_7b/gather_two_data')
    parser.add_argument('--cloud_model_path',type=str,default=r'/data/gm/DoubleCap/models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--edge_model_path', nargs='+', type=str,
                        default=[
                             r'/data/gm/DoubleCap/models/Qwen2-VL-2B-Instruct',
                             r'/data/gm/DoubleCap/models/Qwen2.5-VL-3B-Instruct'
                        ])





    parser.add_argument('--client_num', type=int, default=2, help='number of users')
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--peft', type=str, default='perTucker',choices=['lora', 'perTucker','Tlora'])
    parser.add_argument('--cloud_peft', type=str, default='Tlora', choices=['lora', 'Tlora','perTucker'])
    parser.add_argument('--aggregation', type=str, default='doublecap')

    parser.add_argument('--use_module', nargs='+', type=bool, default=[True,True,True], help="[pertucker,d2cloud,d2edge]")


    parser.add_argument('--gpa_hidden_size',type=int,default=128)

    parser.add_argument('--seed_num', type=int, default=42)
    parser.add_argument('--result_dir_name', type=str, default='result')
    parser.add_argument('--startC', type=int, default=0, help="number of rounds of training")
    parser.add_argument('--cloud_epochs', type=int, default=4, help="number of rounds of training")
    parser.add_argument('--epochs', type=int, default=1, help="number of rounds of training")
    parser.add_argument('--communications', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--edge_bs', nargs='+',type=int, default=[1, 1, 1], help="[edge0,edge1,cloud]")
    parser.add_argument('--gcs', type=int, default=2, help="gradient_accumulation_steps")
    parser.add_argument('--gpus', type=int, default=7)

    parser.add_argument('--edge_lr', nargs='+',type=float, default=[5e-2, 5e-2], help="[edge0,edge1]")
    # parser.add_argument('--edge_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--d2cloud_lr', type=float, default=5e-2, help='learning rate')
    parser.add_argument('--d2edge_lr', type=float, default=5e-2, help='learning rate')

    parser.add_argument('--edge_checkpoint-frequency', default=500, type=int, help='Frequency of showing example outputs')
    parser.add_argument('--lambda_lm', type=float, default=0.01, help='learning rate')
    parser.add_argument('--edge_alpha', type=float, default=1, help='learning rate')
    parser.add_argument('--d2cloud_alpha', type=float, default=0.01, help='learning rate')
    parser.add_argument('--ib_w', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--vib_w', type=float, default=0.1, help='learning rate')
    parser.add_argument('--discriminator_w', type=float, default=0.6, help='learning rate')
    parser.add_argument('--struct_w', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--logit_w', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--ce_w', type=float, default=1, help='learning rate')
    parser.add_argument('--T', type=float, default=1, help='learning rate')



    parser.add_argument('--local_rank', type=int, default=-1, help='Used by deepspeed')
    parser = deepspeed.add_config_arguments(parser)






    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    argsDict = args.__dict__


    return args,argsDict
