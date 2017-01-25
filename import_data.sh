#!/bin/bash

## import surface data
for f in $(ls /archive/alumni/anne/current/PNC/FC_analysis/func_on_surface_smooth/6*.dset); do
    filename=$(basename ${f})
    rsync ${f} data/${filename}
done

## import maget segmentation
#prefix='PNC_'
#for f in $(ls /archive/alumni/anne/current/PNC/FC_analysis/AMY_masks_ns_resampled_nii/*.nii); do
#    filename=$(basename ${f})
#    filename=${filename#$prefix}
#    rsync ${f} data/${filename}
#done

# import volume data
for d in $(ls -d /external/PNC/data/rsfc-awheel/data/PNC/*/); do
    subject_prefix=$(basename ${d})
    filename=$(ls ${d}/REST/SESS01/func_volsmooth*.nii.gz)
    rsync ${filename} data/${subject_prefix}_func_volsmooth.nii.gz
done
