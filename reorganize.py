import shutil
import os
from tqdm import tqdm

thuman_path = 'data/THuman'
save_path = 'data/humanscan_wbg'
output_path = os.path.join(save_path, 'human_train')
item_thuman = os.listdir(os.path.join(thuman_path, 'THuman2.0_Release'))
if not os.path.exists(output_path):
    os.mkdir(save_path)
    os.mkdir(output_path)

for item in tqdm(item_thuman):
    item_folder = os.path.join(thuman_path, 'THuman2.0_Release', item, '18views_3', 'render')
    # item_folder_norm = os.path.join(thuman_path, 'THuman2.0_Release', item, '54views_wbg', 'normal')
    item_folder_pose = os.path.join(thuman_path, 'THuman2.0_Release', item, '18views_3', 'calib')
    if not os.path.exists(os.path.join(output_path, item)):
        os.mkdir(os.path.join(output_path, item))
        os.mkdir(os.path.join(output_path, item, 'smplx'))
    shutil.copy(os.path.join(thuman_path, 'THUman20_Smpl-X', item, 'smplx_param.pkl'), os.path.join(output_path, item, 'smplx'))
    shutil.copytree(item_folder, os.path.join(output_path, item, 'rgb'))
    # shutil.copytree(item_folder_norm, os.path.join(output_path, item, 'norm'))
    shutil.copytree(item_folder_pose, os.path.join(output_path, item, 'pose'))

