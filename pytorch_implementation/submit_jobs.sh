#$ -S /bin/bash 
#$ -N output
#$ -o /ifs/loni/faculty/thompson/four_d/vgupta/Tools/Remove_GadContrast/pytorch_implementation/logs


py_dir=/ifs/loni/faculty/thompson/four_d/vgupta/Tools/Softwares/miniconda3/bin
apps_dir=/ifs/loni/faculty/thompson/four_d/vgupta/Tools/Remove_GadContrast/pytorch_implementation

base_dir=/ifs/loni/faculty/thompson/four_d/vgupta/Data/Gad_data/rawdicoms/numpy_data/
num_epochs=3
model_name=MLP
model_id=3
activation=relu
opt_flag=SGD
momentum=0.9
data_flag=temp
<<COMMENT
batch_size=${batch_size}
step_size=${step_size}
loss_flag=${loss_flag}
COMMENT
batch_size=20
step_size=0.0001
loss_flag=KL

out_dir=${base_dir}MLP_PyTorch/Loss_${loss_flag}_Optimization_${opt_flag}_step_size_${step_size}_batch_size_${batch_size}_activation_${activation}/

mkdir -p ${out_dir}
$py_dir/python3.5 ${apps_dir}/MLP_v2.py $base_dir $out_dir ${batch_size} ${num_epochs} $model_name $model_id $step_size $activation $opt_flag ${momentum} ${loss_flag} ${data_flag}
