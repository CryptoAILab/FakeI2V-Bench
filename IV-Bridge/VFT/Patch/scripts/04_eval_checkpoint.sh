# example: python test_runs.py checkpoints/temp-inverted-and-unpaired_seed0_xception_block5_constant_p10/ gen_models val

# model arch and dataset
# for expt in checkpoints/gp1-* checkpoints/gp1[a-d]-*
for expt in checkpointsVFT/gp3-faceforensics-nt_seed0_xception_block5_constant_p10
do
for part in test # val
do
    cmd="python test_runs.py $expt IV-Bridge $part"
    echo $cmd
    eval $cmd
done
done


