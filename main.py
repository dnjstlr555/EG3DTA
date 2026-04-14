import torch
import random
import numpy as np

from mmcv.cnn.utils import get_model_complexity_info
from argparse import ArgumentParser

from train import *
from models.eg3dta import EG3DTA

def construct_cfg(args):
    cfg = {
        'in_channels': args.in_channels,
        'base_channels': args.base_channels,
        'num_gcn_scales': args.num_gcn_scales,
        'num_g3d_scales': args.num_g3d_scales,
        'num_person': args.num_person,
        'tcn_dropout': args.tcn_dropout,
        'input_size': args.input_size,
        'out_size': args.out_size,
        'graph_type': args.graph_type
    }
    return cfg

def model_init(model, lr=0.01):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    criterion = nn.HuberLoss(delta=0.8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9999,9999], gamma=0.5)
    return optimizer, criterion, scheduler

def new_model(cfg, device):
    model = EG3DTA(
        in_channels=cfg['in_channels'],
        base_channels=cfg['base_channels'],
        num_gcn_scales=cfg['num_gcn_scales'],
        num_g3d_scales=cfg['num_g3d_scales'],
        num_person=cfg['num_person'],
        tcn_dropout=cfg['tcn_dropout'],
        input_size=cfg['input_size'],
        out_size=cfg['out_size'],
        graph_type=cfg['graph_type']
    )
    return model.to(device)

def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def permute_pyskl(x):
    #train_x, train_y, test_x, test_y, valid_x, valid_y, stdy
    #train x -> [N, T,V, C] -> [N, C, T, V, 1]
    train_x, train_y, test_x, test_y, valid_x, valid_y, stdy = x
    train_x = train_x.permute(0, 4, 2, 3, 1)
    test_x = test_x.permute(0, 4, 2, 3, 1)
    valid_x = valid_x.permute(0, 4, 2, 3, 1)
    return train_x, train_y, test_x, test_y, valid_x, valid_y, stdy

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args):
    cfg = construct_cfg(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Test run with device: {device}")
    gcn = new_model(cfg, device)
    inp = torch.randn(3, 1, 100, 25 if cfg['graph_type'] == "nturgb+d" else 39, 3).to(device)
    with torch.no_grad():
        out = gcn(inp)

    flops, params = get_model_complexity_info(gcn, (1, 100, 25 if cfg['graph_type'] == "nturgb+d" else 39, 3), as_strings=True, print_per_layer_stat=False)
    print(f"FLOPs: {flops}, Params: {params}")
    
    seed_all(23)
    train_kimore(args.data_path, model_init, lambda:new_model(cfg, device), 
                 prefix="kimore" if 'kimore' in args.data_path else "uiprmd", 
                 lrs=0.01,
                 batch_size=32, 
                 epochs=800, 
                 ex_only=[1,2,3,4,5] if 'kimore' in args.data_path else [1,2,3,4,5,6,7,8,9,10], 
                 raw_data_func=lambda x:x,
                 processed_data_func=permute_pyskl,
                 earlystop_patient=300, 
                 saveonperiod=False, 
                 savetestlabel=False,
                 device_sp = device,
                 inference_only=True if args.phase == "eval" else False)

def main():
    parser = ArgumentParser(description="Train and evaluate a model on the dataset")
    parser.add_argument("--data_path", type=str, default="./data/kimore_kfold_norm.pkl", help="Path to the dataset")
    parser.add_argument("--phase", type=str, choices=["train", "eval"], default="train", help="Whether to train or evaluate the model")

    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--base_channels", type=int, default=24, help="Number of base channels in the model")
    parser.add_argument("--num_gcn_scales", type=int, default=13, help="Number of GCN scales")
    parser.add_argument("--num_g3d_scales", type=int, default=6, help="Number of G3D scales")
    parser.add_argument("--num_person", type=int, default=1, help="Number of persons in the input data")
    parser.add_argument("--tcn_dropout", type=float, default=0, help="Dropout rate for TCN layers")
    parser.add_argument("--input_size", type=int, default=100, help="Input sequence length")
    parser.add_argument("--out_size", type=int, default=1, help="Output size")
    parser.add_argument("--graph_type", type=str, choices=["nturgb+d", "vicon"], default="nturgb+d", help="Type of graph structure to use in the model")

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()