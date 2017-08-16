
batch_size_arr=( 1000 2000  3000 )
step_size_arr=( 0.0001 0.001 0.005 )
activation_arr=( 'tanh' 'relu' 'sigmoid' )
loss_arr=( 'SSD' 'KL' 'BCE' )

#apps_dir=/home/rcf-proj2/vg/Softwares/Remove_GadContrast/keras_implementation

apps_dir=/ifs/loni/faculty/thompson/four_d/vgupta/Tools/Remove_GadContrast/pytorch_implementation

for i in "${batch_size_arr[@]}"
do
    for j in "${step_size_arr[@]}"
    do
      for k in "${loss_arr[@]}"
      do
      #  for m in `seq 3 5`; do
            qsub -v loss_flag=$k,step_size=$j,batch_size=$i ${apps_dir}/submit_jobs.sh
      #      echo $mi
            echo $i $j  $k
      #  done
      done
    done
done
