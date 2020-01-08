#!/bin/bash

today=`date +'%y_%m_%d_%H_%M_%S'`
mkdir -p log
mkdir -p log/$today
make allgemm

#For CUDA10 GPU boost
#for(( x =0 ; x <  `nvidia-smi| grep N/A |wc -l` ; x++))
#do
#       GName=`nvidia-smi -q -i $x |grep "Product Name" | cut -d ":" -f2 |cut -d " " -f3`
#       Memory=`nvidia-smi  -i $x -q | grep Memory | grep Hz | tail -n 1 | cut -d ":" -f2 | cut -d " " -f2`
#       Graphic=`nvidia-smi  -i $x -q | grep Graphic | head -n 4 | tail -n 1 | cut -d ":" -f2 | cut -d " " -f2`
#       echo "Setup : $GName-> Application clock= $Memory,$Graphic"
#       nvidia-smi -i $x -ac  $Memory,$Graphic
#done
#ubuntu
#apt-get install -y linux-tools-$(uname -r)
#RHEL7
#yum install kerne-tools -y
#cpupower frequency-info
#cpupower frequency-set --governor performance
#cpupower monitor


GHZ=`cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq`

#CPU BOOST
echo "Set CPU Freq to : $GHZ"
for ((x=0;x<$(lscpu  | grep On-line | lscpu  | grep On-line |  awk '{print $4}'| cut -d "-" -f2
 | cut -d "-" -f2);x++))
do
 echo  $GHZ > /sys/devices/system/cpu/cpu${x}/cpufreq/scaling_min_freq
 echo  $GHZ > /sys/devices/system/cpu/cpu${x}/cpufreq/scaling_max_freq
done


#Test Script
for(( x =0 ; x <  `nvidia-smi| grep N/A |wc -l` ; x++))
do
	echo "single thread for each GPU"
	# Only run FP16 mul with FP16 accumulate on Tensorcore
#	CUDA_VISIBLE_DEVICES=$x ./bin/allgemm FP16_TENSOR | tee log/$today/allgemm_FP16_TENSOR_GPU${x}_${today}.csv  &
	# Only run FP16 mul with FP32 accumulate on Tensorcore
	#CUDA_VISIBLE_DEVICES=$x ./bin/allgemm FP16_32_TENSOR | tee log/$today/allgemm_FP16_32_TENSOR_GPU${x}_${today}.csv &
	# Only run FP32 mul with FP32
	#CUDA_VISIBLE_DEVICES=$x ./bin/allgemm FP32_CUDA | tee log/$today/allgemm_FP32_CUDA_GPU${x}_${today}.csv &
	# Only run FP16 mul with FP16
	#CUDA_VISIBLE_DEVICES=$x ./bin/allgemm FP16_CUDA | tee log/$today/allgemm_FP16_CUDA_GPU${x}_${today}.csv &
done




#Test Script for openmp
echo "multiple thread for each GPU"
# Only run FP16 mul with FP16 accumulate on Tensorcore
./bin/allgemm_omp FP16_TENSOR | tee log/$today/allgemm_FP16_TENSOR_GPU${x}_${today}.csv  &
# Only run FP16 mul with FP32 accumulate on Tensorcore
#./bin/allgemm_omp FP16_32_TENSOR | tee log/$today/allgemm_FP16_32_TENSOR_GPU${x}_${today}.csv &
# Only run FP32 mul with FP32
#./bin/allgemm_omp FP32_CUDA | tee log/$today/allgemm_FP32_CUDA_GPU${x}_${today}.csv &
# Only run FP16 mul with FP16
#./bin/allgemm_omp FP16_CUDA | tee log/$today/allgemm_FP16_CUDA_GPU${x}_${today}.csv &


