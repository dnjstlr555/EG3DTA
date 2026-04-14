import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from argparse import ArgumentParser
from models.eg3dta import EG3DTA
from train import getdata, preprocess, get_loader

def construct_cfg():
    cfg = {
        'in_channels': 3,
        'base_channels': 24,
        'num_gcn_scales': 13,
        'num_g3d_scales': 6,
        'num_person': 1,
        'tcn_dropout': 0,
        'input_size': 100,
        'out_size': 1,
        'graph_type': "nturgb+d" #adjust this to vicon if you want to test with uiprmd (39 joints)
    }
    return cfg


def new_model(cfg, device, ex, fold):
    model = EG3DTA(
        in_channels=cfg['in_channels'],
        base_channels=cfg['base_channels'],
        num_gcn_scales=cfg['num_gcn_scales'],
        num_g3d_scales=cfg['num_g3d_scales'],
        num_person=cfg['num_person'],
        tcn_dropout=cfg['tcn_dropout'],
        input_size=cfg['input_size'],
        out_size=cfg['out_size'],
        graph_type=cfg['graph_type'],
        return_attn=True
    )
    use_string = 'kimore' if cfg['graph_type'] == "nturgb+d" else 'uiprmd'
    model.load_state_dict(torch.load(f'./results/{use_string}_ex{ex}_fold{fold}_best.pth', map_location=device,), strict=True)
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

def get_complete_data(cfg, ex, fold):
    data_path="./data/uiprmd_kfold.pkl" if cfg['graph_type'] == "vicon" else "./data/kimore_kfold_norm.pkl"
    raw_data=joblib.load(data_path)
    totaldata = getdata(raw_data, ex, cv=True)
    crcvdata = [d[fold] for d in totaldata]
    train_x, train_y, test_x, test_y, _, __, stdy = permute_pyskl(preprocess(crcvdata))
    train_loader, test_loader, valid_loader = get_loader(train_x, train_y, test_x, test_y, test_x, test_y, batch_size=16)
    return train_loader, test_loader, valid_loader, stdy, crcvdata[2]

def plot_skeleton(frame, joint_imp, ax, size_factor=1):
    neighbor_base = [
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
        (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
        (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
    ] if frame.shape[0] == 25 else [ #it is impossible to plot vicon data, since the data is not a position data.
        (1, 2), (2, 3), (3, 5), (4, 3),  
        (5, 6), (6, 9), 
        (7, 9), (8, 9), 
        (9, 10),
        (10, 6),
        (11, 7), (12, 11), (13, 12), (14, 13), (15, 14), (16, 15),
        (17, 8), (18, 17), (19, 18), (20, 19), (21, 20), (22, 21), (23, 22), 
        (24, 10), (25, 10),
        (26, 25), (27, 26),
        (28, 24), (29, 28), (30, 29), (31, 30), (32, 31), (33, 31),
        (34, 25), (35, 34), (36, 35), (37, 36), (38, 37), (39, 37) 
    ]
    skeleton_links = [(i - 1, j - 1) for (i, j) in neighbor_base]
    
    for link in skeleton_links:
        x = [frame[link[0], 0], frame[link[1], 0]]
        y = [frame[link[0], 1], frame[link[1], 1]]
        z = [frame[link[0], 2], frame[link[1], 2]]
        ax.plot(x, y, z, 'b', linewidth=1.5, alpha=0.6)
    
    ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], 
               c='r', s=10 * size_factor, alpha=0.8)

    ax.view_init(elev=10, azim=90)
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-1, 0.7])
    ax.set_zlim([-1, 1])
    ax.set_axis_off()

def plot_map_and_skeletons(sequence, joint_imp, attention_map, num_frames=8, name='attention_with_8_skeletons.png'):
    T = sequence.shape[0]
    selected_indices = np.linspace(0, T - 1, num_frames, dtype=int)
    fig = plt.figure(figsize=(20, 4)) 
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[1, 4], wspace=0.05)

    ax_map = fig.add_subplot(gs_main[0])
    im = ax_map.imshow(attention_map, cmap='hot', aspect='equal')
    ax_map.set_box_aspect(1)
    
    ax_map.set_title('', fontsize=12, fontweight='bold')
    ax_map.set_xlabel('Frame Index T', fontsize=10)
    ax_map.set_ylabel('', fontsize=10)
    
    ax_map.set_xticks(selected_indices)
    ax_map.set_yticks(selected_indices)
    ax_map.tick_params(axis='both', labelsize=8)

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    gs_skel = gridspec.GridSpecFromSubplotSpec(1, num_frames, subplot_spec=gs_main[1], wspace=0.0)
    
    for i, idx in enumerate(selected_indices):
        ax3d = fig.add_subplot(gs_skel[i], projection='3d')
        
        plot_skeleton(sequence[idx], joint_imp[idx], ax3d, size_factor=1)
        ax3d.set_title(f't={idx}', fontsize=10)
        ax3d.set_box_aspect([1, 1.5, 2])

    plt.tight_layout()
    print(f"Saving static plot to {name}...")
    plt.savefig(name, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    plt.close(fig)
    print("Done.")


def visualize(args):
    cfg = construct_cfg()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Test run with device: {device}")
    gcn = new_model(cfg, device, ex=args.ex, fold=args.fold)
    inp = torch.randn(3, 1, 100, 25 if cfg['graph_type'] == "nturgb+d" else 39, 3).to(device)
    with torch.no_grad():
        out, attn = gcn(inp)
    print("Test run successful. Output shape:", out.shape)
    train_loader, test_loader, valid_loader, stdy, test_x = get_complete_data(cfg, ex=args.ex, fold=args.fold)
    gcn.eval()
    inputs = []
    attns = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            output, attn = gcn(data)
            work = attn[2].cpu().detach().numpy()
            inp = data.cpu().detach().numpy()
            inputs.append(inp)
            attns.append(work)
    attns = np.concatenate(attns,axis=0)
    inputs = np.concatenate(inputs,axis=0)

    #mkdir "results/visualization" if not exists
    os.makedirs("results/visualization", exist_ok=True)
    plt.clf()
    for i in range(attns.shape[0]):
        if i<30: continue
        if i>60: break
        toplot = attns[i]  # Shape: (V, num_heads, T, T)
        joint_importance_over_time = np.mean(toplot, axis=(1, 2)).transpose()
        attention_map_2d = np.mean(toplot[8], axis=0) 
        plot_map_and_skeletons(
            sequence=test_x[i], 
            joint_imp=joint_importance_over_time, 
            attention_map=attention_map_2d,
            num_frames=7,
            name=f"results/visualization/ex{args.ex}_fold{args.fold}_sample{i}_attention.png"
        )
    print("Visualization saved in results/visualization/")

if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--ex", type=int, default=1, help="Experiment number")
    argument_parser.add_argument("--fold", type=int, default=0, help="Fold number")
    args = argument_parser.parse_args()
    visualize(args)