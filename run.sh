#!/bin/bash 
#SBATCH --job-name=md                                                                                                                                                     
#SBATCH --output=md.out                                                                                                                                                   
#SBATCH --error=md.err                                                                                                                                                    
#SBATCH -A bsavoie                                                                                                                                                                                                                                                                                                               
#SBATCH -p cpu                                                                                                                                                            
#SBATCH --nodes=1                                                                                                                                                         
#SBATCH --ntasks-per-node=64                                                                                                                                            
#SBATCH --time=5:00:00           

module purge
module load intel    

# Check if mpirun is now found before launching
which mpirun

mpirun -np 64 /depot/bsavoie/apps/lammps/exe/lmp_mpi_190322 -in test.in