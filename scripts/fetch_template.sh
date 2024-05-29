#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

mkdir -p lib/models/deformers/smplx/SMPLX

# username and password input
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de/, according to Installation Instruction."
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)

# SMPLX 
echo -e "\nDownloading SMPL-X model..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O 'models_smplx_v1_1.zip' --no-check-certificate --continue
unzip models_smplx_v1_1.zip -d lib/models/deformers/smplx/SMPLX
mv lib/models/deformers/smplx/SMPLX/models/smplx/* lib/models/deformers/smplx/SMPLX
rm -rf lib/models/deformers/smplx/SMPLX/models
rm -f models_smplx_v1_1.zip

mkdir -p work_dirs/cache/template

cd work_dirs/cache/template
echo -e "\nDownloading SMPL-X segmentation info..."
wget https://github.com/Meshcapade/wiki/blob/main/assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json

echo -e "\nDownloading SMPL-X UV info..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_uv.zip' -O './smplx_uv.zip' --no-check-certificate --continue
unzip smplx_uv.zip -d ./smplx_uv
mv smplx_uv/smplx_uv.obj ./
rm -f smplx_uv.zip
rm -rf smplx_uv

echo -e "\nDownloading SMPL-X FLAME correspondence info..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_mano_flame_correspondences.zip' -O './smplx_mano_flame_correspondences.zip' --no-check-certificate --continue
unzip smplx_mano_flame_correspondences.zip -d ./smplx_mano_flame_correspondences
mv smplx_mano_flame_correspondences/SMPL-X__FLAME_vertex_ids.npy ./
rm -f smplx_mano_flame_correspondences.zip
rm -rf smplx_mano_flame_correspondences

echo -e "\nDownloading FLAME template from neural-head-avatars repo..."
wget https://raw.githubusercontent.com/philgras/neural-head-avatars/main/assets/flame/head_template_mesh_mouth.obj

echo -e "\nDownloading FLAME template from DECA repo..."
wget https://raw.githubusercontent.com/yfeng95/DECA/master/data/head_template.obj

echo -e "\nYou need to register at http://flame.is.tue.mpg.de/, according to Installation Instruction."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading FLAME segmentation info..."
wget 'https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip' -O 'FLAME_masks.zip' --no-check-certificate --continue
unzip FLAME_masks.zip -d ./FLAME_masks
mv FLAME_masks/FLAME_masks.pkl ./
rm -f FLAME_masks.zip
rm -rf FLAME_masks

cd ../../..

echo -e "\n Finish"





