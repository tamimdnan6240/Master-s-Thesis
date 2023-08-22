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

epochs=30

for epoch in "${epochs[@]}"
do
      srun --nodes=1 --ntasks=2 --cpus-per-task=$SLURM_CPUS_PER_TASK python UNet_thesis_EdmCrack.py "$epoch" > output_epoch_${epoch}.txt & # launch each inner loop iteration as a separate task in the background
done

wait # wait for all the tasks to complete before exiting
