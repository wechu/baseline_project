#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --cpus-per-task=1       # CPU cores/threads
#SBATCH --mem=128mb             # memory per node
#SBATCH --time=10:59:00            # time (HH:MM:SS)
#SBATCH --array=0-77   # change this to match the number of job_i files that you have

module load python/3.6
module load scipy-stack

echo "Starting run at: `date`"

echo "Starting task $SLURM_ARRAY_TASK_ID"
FILE=jobs/job_$SLURM_ARRAY_TASK_ID.txt
cat $FILE | while read line
do
 # execute line in file
 $line
done

echo "Program test finished with exit code $? at: `date`"