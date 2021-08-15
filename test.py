import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import ModifiedRTTestDataset, DavisTransform
from model_R34 import Interactive
import torch.nn.functional as F
from imageio import imwrite
import random
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    setup_seed(1024)
    model_dir = "./saved_model/"
    batch_size_val = 1
#    dataset = "/local/riemann/home/msiam/DAVIS/"
#    transform = DavisTransform(384, int(384 * 1.75))
#    DAVIS_dataset = ModifiedRTTestDataset(dataset, 2, file_list='ImageSets/480p/val.txt',
#                                          img_prefix="JPEGImages/480p/", transform_fn=transform)
    dataset = "/local/riemann/home/msiam/MoCA_filtered2/"
    transform = DavisTransform(384, int(384 * 1.75))
    DAVIS_dataset = ModifiedRTTestDataset(dataset, 2, file_list='val.txt',
                                          img_prefix="JPEGImages/", transform_fn=transform)

    DAVIS_dataloader = DataLoader(DAVIS_dataset, batch_size=1, shuffle=False, num_workers=0)
    net = Interactive().cuda()
    model_name = 'model_R34.pth'
    ckpt = torch.load(model_dir + model_name)['state_dict']
    model_dict = net.state_dict()
    pretrained_dict = {k[7:]: v for k, v in ckpt.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.eval()

    if not os.path.exists('results'):
        os.mkdir('results')
    for data in tqdm(DAVIS_dataloader):
        img, fw_flow, bw_flow, label_org = data['video'].cuda(), data['fwflow'].cuda(), data['bwflow'].cuda(), data['label_org'].cuda()
        _, _, _, H, W = label_org.size()
        flow = torch.cat((fw_flow, bw_flow), 2)
        with torch.no_grad():
            out, _ = net(img, flow)
        out = F.interpolate(out[0], (H, W), mode='bilinear', align_corners=True)
        out = out[0, 0].cpu().numpy()
        out = (out - np.min(out) + 1e-12) / (np.max(out) - np.min(out) + 1e-12) * 255.
        out = out.astype(np.uint8)
        fname = data['name'][0][0] + '/' + data['name'][1][0].replace('jpg', 'png')
        if not os.path.exists('results/' + data['name'][0][0]):
            os.makedirs('results/' + data['name'][0][0])
        imwrite("./results/"+fname, out)
