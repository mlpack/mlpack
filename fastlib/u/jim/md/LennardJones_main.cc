/**
 * @file LennardJones_main.cc
 * 
 * This program creates an instance of the LennardJones problem
 * class, and updates the velocities using a leapfrogging scheme
 * until a specified end time is reached. 
 *
 * @see LennardJones.h
 */


#include "LennardJones.h"

int main(int argc, char *argv[])
{
  fx_init(argc, argv);
  const char* fp;
  fp = fx_param_str(NULL, "data", "default.txt");
  
  Matrix atom_mat;
  FILE *tree, *naive;
  double time = 0;
  double eps, sig, mass, cutoff, time_step, stop_time;
  LennardJones simulation;
  LennardJones naive_test;  
  
  // Initialize with default values:
  // Values are in eV, Angstroms, and amu.
  // this places time (roughly) in units of nanoseconds
  // Default values correspond to Argon, taken from Ashcroft & Mermin
  eps = fx_param_double(0, "eps", 0.0104);
  sig = fx_param_double(0, "sig", 2.74);
  mass = fx_param_double(0, "m", 40);
  time_step = fx_param_double(0, "dt", 1.0e-3);
  stop_time = fx_param_double(0, "tf", 1.0e0);
  cutoff = fx_param_double(0, "r_max", 6*sig);
  cutoff = cutoff*cutoff;

  naive = fopen("out_naive.dat", "w+");
  tree = fopen("out_tree.dat", "w+");

  // Read Atom Matrix
  data::Load(fp, &atom_mat);


  /**
   * Begin simulation, and run to end time.
   */
  simulation.Init(atom_mat, eps, sig, mass, cutoff);
  simulation.UpdateVelocities(time_step/2);
  
  naive_test.InitNaive(atom_mat, eps, sig, mass);
  naive_test.UpdateVelocitiesNaive(time_step/2);
  
  while (time < stop_time){
    simulation.UpdateVelocities(time_step);
    simulation.UpdatePositions(time_step);
    
    naive_test.UpdatePositionsNaive(time_step);
    naive_test.UpdateVelocitiesNaive(time_step);
    
    time = time + time_step;
  
  }

  /**
   * Record final positions according to both methods.
   */
  simulation.WritePositions(tree);
  naive_test.WritePositions(naive);

  fx_done();
  fclose(naive);
  fclose(tree);

}
