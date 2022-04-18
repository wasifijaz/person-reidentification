#!/bin/bash

. ~/.bashrc
module load anaconda3/5.3.1

conda activate PPUU
conda install -n PPUU nb_conda_kernels

chikka=$1

echo $chikka


cd 

python hyper_supervise_validation.py --dataset="prid_subset" >> ~/code/Personre-id-master/outputs/prid_diff_val_map.out
python config_trainer.py --focus=map --dataset=prid --opt=$chikka --name=_prid_cl_centers_ --cl-centers >>  ~/code/official/output/prid_cl_centers_$chikka.out
python bagoftricks.py --pretrained-model="/beegfs/pp1953/ResNet50ta_bt_mars_cl_centers__8__checkpoint_ep181.pth.tar" -d="prid" --opt=$chikka --name="_prid_CL_CENTERS_" --validation-training --cl-centers --print-freq=10
python3 config_trainer.py --focus=map --dataset=prid --opt=38 --name=_prid_cl_centers_ --cl-centers >> output/prid_cl_centers_38.out