#!/bin/sh

#SBATCH --job-name=tamimadnan    # create a short name for your job
#SBATCH --partition=Andromeda        # partition 
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=2               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=30gb         # memory per cpu-core (4G is default)
#SBATCH --time=48:00:00          # maximum time needed (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=tadnan@uncc.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

## the above is multithreaded job 

module load pytorch

epochs=(10 20 30 40 50 100 500 1000)

optimizers=("Adam" "SGD") # add the optimizers to be used as an array

for epoch in "${epochs[@]}"
do
  for optimizer in "${optimizers[@]}" # add a loop for the optimizers
  do
      srun --ntasks 1 --cpus-per-task $SLURM_CPUS_PER_TASK python ResNet50_tranfered_pavement.py "$epoch" "$optimizer"> output_epoch_optimizers_${epoch}_${optimizer}.txt & # launch each inner loop iteration as a separate task in the background
  done
done

wait # wait for all the tasks to complete before exiting
