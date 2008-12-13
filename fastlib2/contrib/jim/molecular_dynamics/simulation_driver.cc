/**
 * @file simulation_driver.cc
 * 
 * @author Jim Waters (jwaters6@gatech.edu)
 *
 * This program creates an instance of the LennardJones problem
 * class, and updates the velocities using a leapfrogging scheme
 * until a specified end time is reached. 
 *
 * @see lennard_jones.h
 */

#include "multi_physics_system.h"
#include "particle_tree.h"
#include "raddist.h"

#define PI 3.14159265358979
#define K_B  8.63e-5  // In eV / Kelvin


const fx_entry_doc root_entries[] = { 
  {"dt", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies time step of dynamic simulation. \n"},
  {"tf", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies duration of simulation \n"},
  {"temp", FX_PARAM, FX_DOUBLE, NULL,
   "Temperature of simulation. \n"},
  {"pos", FX_REQUIRED, FX_STR, NULL, 
   "Kinematic Information of particles \n"},
  {"lj", FX_REQUIRED, FX_STR, NULL, 
   "Parameters of two-body potential function \n"},
  {"rad", FX_PARAM, FX_STR, NULL, 
   "Name of radial distribution output \n"},
  {"coord", FX_PARAM, FX_STR, NULL, 
   "Name of coordinate output file \n"},
  {"stats", FX_PARAM, FX_STR, NULL, 
   "Name of stats output file \n"},
  FX_ENTRY_DOC_DONE  
};



const fx_submodule_doc md_submodules[] = {
  {"param", &param_doc, "Parameters for MD-simulation \n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc root_doc = {
  root_entries, md_submodules, 
  "Simulation Parameters \n"
};

int main(int argc, char *argv[])
{
  fx_module *root = fx_init(argc, argv, &root_doc);
  const char* fp_k;
  const char* fp_l;  
  const char* fp_three;
  const char* fp_stats;
  const char* fp_coords;
  const char* fp_rad;

  FILE *coords;
  FILE *stats; 
  FILE *radial_distribution;
 
  
 
  double time_step, stop_time, time;
  
  // Input files
  fp_k = fx_param_str_req(NULL, "pos");
  fp_l = fx_param_str_req(NULL, "lj");
  fp_three = fx_param_str_req(NULL, "three");
  
  // Output Files
  fp_stats = fx_param_str(NULL, "stats", "tree_stats.dat");
  fp_rad = fx_param_str(NULL, "rad", "raddist.dat");
  fp_coords = fx_param_str(NULL, "coord", "coords.dat");

  coords = fopen(fp_coords, "w+");
  stats = fopen(fp_stats, "w+");  
  radial_distribution = fopen(fp_rad, "w+");
  Matrix atom_matrix, lj_matrix, at_matrix;
 
  struct datanode* parameters = fx_submodule(root, "param");

  time_step = fx_param_double(0, "dt", 1.0e0);
  stop_time = fx_param_double(0, "tf", 1.0e2); 
  double set_temp = fx_param_double(0, "temp", -1.0);
  printf("Set Temperature: %f \n", set_temp);
  set_temp = set_temp * (3.0*K_B);

  // Read Atom Matrix
  data::Load(fp_k, &atom_matrix);
  Vector signs_, powers_;  
  data::Load(fp_l, &lj_matrix);
  data::Load(fp_three, &at_matrix);
  int n_atoms = lj_matrix.n_cols();
  Vector temp;
  lj_matrix.MakeColumnVector(n_atoms-2, &temp);
  powers_.Init(lj_matrix.n_rows());
  powers_.CopyValues(temp);
  temp.Destruct();
  lj_matrix.MakeColumnVector(n_atoms-1, &temp);
  signs_.Init(lj_matrix.n_rows());
  signs_.CopyValues(temp);
  lj_matrix.ResizeNoalias(n_atoms - 2);

  Vector use_dims;
  use_dims.Init(3);
  use_dims[0] = 0;
  use_dims[1] = 1;
  use_dims[2] = 2;

  fx_timer_start(parameters, "Tree_Based");
  MultiPhysicsSystem simulation;
  printf("\n------------------\nTree Simulation \n------------------ \n");
 
  simulation.Init(atom_matrix, parameters);
  simulation.InitStats(lj_matrix, signs_, powers_);
  simulation.InitAxilrodTeller(at_matrix);  
  printf("Finished Initialization. Updating Momentum. \n");
  simulation.UpdateMomentum(time_step / 2);
  time = 0;  
  double target_pct = 0.9;
  int target_trips = 0;

  RadDist tree_simulation;
  tree_simulation.Init(450, 15.0);
  tree_simulation.WriteHeader(radial_distribution);

  double temperature, diffusion, pressure = 0;
  while (time < stop_time){
    double pct = simulation.GetPercent();
    int trips = simulation.GetTrips();
    if (unlikely(time < 2*time_step)){
      target_pct = pct;
      target_trips = trips;
    }
    simulation.UpdatePositions(time_step);   
    if (pct < 0.85*target_pct || trips > 1.1*target_trips){
      simulation.RebuildTree();
      simulation.ReinitStats(lj_matrix);     
    }
    if ((int)(time / time_step) % 5 == 1){
      tree_simulation.Reset();
      simulation.RadialDistribution(&tree_simulation);
      tree_simulation.Write(radial_distribution);
      printf("\n Time: %f \n-------------\n", time);
      temperature = simulation.ComputeTemperature();
      temperature = temperature / (3.0*K_B);
      printf("Temperature: %f \n", temperature);
      pressure = simulation.ComputePressure();
      printf("Pressure: %f \n", pressure);
      diffusion = simulation.ComputeDiffusion(atom_matrix);
      printf("Diffusion: %f \n", diffusion);
      printf("Percent Pruned: %f \n", pct);
      printf("Triples Computed: %d \n", trips);
      fprintf(stats, "%f %f %f, %f \n", time, diffusion, pressure,
           temperature);
 
      if (set_temp > 0){
	simulation.ScaleToTemperature(set_temp);
      }
    }    
    
    simulation.UpdateMomentum(time_step);
    pct = simulation.GetPercent();
    time = time + time_step;
  }
  fx_timer_stop(parameters, "Tree_Based");
 
  
  simulation.WriteData(coords);   
  fclose(coords);
  fclose(stats);
  fclose(radial_distribution);
  
  fx_done(root);

}
