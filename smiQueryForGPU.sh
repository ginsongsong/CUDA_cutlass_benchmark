#!/bin/bash
nvidia-smi --query-gpu=index,timestamp,pstate,power.draw,clocks.gr,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 1 

#watch -n 1 nvidia-smi --query-gpu=index,timestamp,pstate,power.draw,clocks.gr,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv 
