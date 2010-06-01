/**
 * @file LennardJones_main.cc
 * 
 * @author Jim Waters (jwaters6@gatech.edu)
 *
 * This program creates an instance of the LennardJones problem
 * class, and updates the velocities using a leapfrogging scheme
 * until a specified end time is reached. 
 *
 * @see lennard_jones.h
 */


#include "lennard_jones.h"

int main(int argc, char *argv[])
{
  fx_module *root = fx_init(argc, argv, NULL);
  const char* fp;
  fp = fx_param_str(NULL, "data", "default.txt");
  
  Matrix atom_matrix;
  FILE *tree_file;
  double time = 0, time_step, stop_time;
  bool do_naive;
  LennardJones simulation;
  struct datanode* parameters = fx_submodule(root, "param");

  // Read Atom Matrix
  data::Load(fp, &atom_matrix);
  
  time_step = fx_param_double(0, "dt", 1.0e-3);
  stop_time = fx_param_double(0, "tf", 1.0e0);
  do_naive = fx_param_bool(0, "check", 0);
 
  tree_file = fopen("out_tree.dat", "w+");

  fx_timer_start(NULL, "Tree_Based");
  // Begin simulation, and run to end time.   
  simulation.Init(atom_matrix, parameters);
  simulation.UpdateVelocities(time_step/2);
  while (time < stop_time){
    simulation.UpdateVelocities(time_step);
    simulation.UpdatePositions(time_step);  
    time = time + time_step;
  }
  fx_timer_stop(NULL ,"Tree_Based");

  // Record final positions according to both methods.   
  simulation.WritePositions(tree_file);
  
  // Record Naive data, compare to tree-base
  if (do_naive){
    LennardJones naive_test;
    FILE *naive_file;
    time = 0;
    naive_file = fopen("out_naive.dat", "w+");

    // Initialize and advance naive simulation
    Matrix naive_atom_matrix; 
    naive_atom_matrix.Copy(atom_matrix);
    naive_test.InitNaive(naive_atom_matrix, parameters);
    naive_test.UpdateVelocities(time_step/2);
    while(time < stop_time){ 
      naive_test.UpdatePositions(time_step);
      naive_test.UpdateVelocities(time_step);
      time = time + time_step;
    }

    // Record results and compare to tree simulation
    naive_test.WritePositions(naive_file);
    fclose(naive_file);
    simulation.CompareToNaive(naive_atom_matrix);
  }

  fx_done(root);

  fclose(tree_file);

}
