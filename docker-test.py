from models.eg3dta import EG3DTA
import torch

def construct_cfg():
    cfg = {
        'in_channels': 3,
        'base_channels': 24,
        'num_gcn_scales': 13,
        'num_g3d_scales': 6,
        'num_person': 1,
        'tcn_dropout': 0.5,
        'input_size': 100,
        'out_size': 1,
        'graph_type': "nturgb+d"
    }
    return cfg


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

def test():
    cfg = construct_cfg()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Test run with device: {device}")
    gcn = new_model(cfg, device)
    inp = torch.randn(3, 1, 100, 25 if cfg['graph_type'] == "nturgb+d" else 39, 3).to(device)
    with torch.no_grad():
        out = gcn(inp)

    print("Test run successful. Output shape:", out.shape)

if __name__ == "__main__":
    test()