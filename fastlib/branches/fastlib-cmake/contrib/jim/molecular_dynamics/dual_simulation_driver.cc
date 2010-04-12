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

#include "dual_physics_system.h"
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
  {"two", FX_REQUIRED, FX_STR, NULL, 
   "Parameters of two-body potential function \n"},
  {"rad", FX_PARAM, FX_STR, NULL, 
   "Name of radial distribution output \n"},
  {"coord", FX_PARAM, FX_STR, NULL, 
   "Name of coordinate output file \n"},
  {"stats", FX_PARAM, FX_STR, NULL, 
   "Name of stats output file \n"},
  {"info", FX_PARAM, FX_INT, NULL,
   "Toggles off output to screen \n"},
  {"diff", FX_PARAM, FX_STR, NULL,
   "Name of diffusion output file \n"},
  {"snapshots", FX_PARAM, FX_INT, NULL,
   "Number of snapshots for diffusion \n"},
  {"naive", FX_PARAM, FX_BOOL, NULL, 
   "Specifies whether to do naive or tree-based simulation \n"},
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
  const char* fp_stats;
  const char* fp_coords;
  const char* fp_rad;
  const char* fp_diff;

  FILE *coords;
  FILE *stats; 
  FILE *radial_distribution;
  FILE *diff;
 
  double time_step, stop_time, time;
  int diff_tot = fx_param_int(0, "snapshots", 1);
  // Input files
  fp_k = fx_param_str_req(NULL, "pos");
  fp_l = fx_param_str_req(NULL, "two");  

  // Output Files
  fp_stats = fx_param_str(NULL, "stats", "tree_stats_dual.dat");
  fp_rad = fx_param_str(NULL, "rad", "raddist_dual.dat");
  fp_coords = fx_param_str(NULL, "coord", "coords_dual.dat");
  fp_diff = fx_param_str(NULL, "diff", "diffusion_dual.dat");

  bool do_naive = fx_param_bool(NULL, "naive", 0);
  
  coords = fopen(fp_coords, "w+");
  stats = fopen(fp_stats, "w+");  
  radial_distribution = fopen(fp_rad, "w+");
  diff = fopen(fp_diff, "w+");
  Matrix atom_matrix, lj_matrix;
 
  struct datanode* parameters = fx_submodule(root, "param");

  time_step = fx_param_double(0, "dt", 0.1);
  stop_time = fx_param_double(0, "tf", 1.0e3); 
  double set_temp = fx_param_double(0, "temp", -1.0);
 
  set_temp = set_temp * (3.0*K_B);

  int info = fx_param_int(0, "info", 0);

  // Read Atom Matrix
  data::Load(fp_k, &atom_matrix);
  Vector signs_, powers_;  
  data::Load(fp_l, &lj_matrix); 
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

  ArrayList<Matrix> positions;

  fx_timer_start(parameters, "Building Tree");
  DualPhysicsSystem simulation;
  printf("\n------------------\nTree Simulation \n------------------ \n");
 
  if (do_naive){
    simulation.InitNaive(atom_matrix, parameters);
  } else {
    simulation.Init(atom_matrix, parameters);
  }
  simulation.InitStats(lj_matrix, signs_, powers_); 
  fx_timer_stop(parameters, "Building Tree");
  printf("Finished Initialization. Updating Momentum. \n");
  fx_timer_start(parameters, "Tree Based");
  simulation.UpdateMomentum(time_step / 2);
  time = 0;  
  double target_pct = 0.9;
 
  RadDist tree_simulation;
  tree_simulation.Init(450, 15.0);
  tree_simulation.WriteHeader(radial_distribution);

  double delta = 10.0, last_time = -2*delta;
  int diff_count = 0;
  positions.Init(diff_tot);

  double temperature, diffusion = 0,pressure = 0;
  while (time < stop_time){
    if (diff_count < diff_tot & time > last_time + delta){ 
      last_time = time;    
      positions[diff_count].Init(3, atom_matrix.n_cols());
      simulation.RecordPositions(positions[diff_count]);
      diff_count++;
    }
    double pct = simulation.GetPercent();  
    if (unlikely(time < 2*time_step)){
      target_pct = pct;     
    }
    simulation.UpdatePositions(time_step);   
    if (pct < 0.85*target_pct){
      simulation.RebuildTree();
      simulation.ReinitStats(lj_matrix);      
    }
    if ((int)(time / time_step -0.5) % 5 == 0){      
      tree_simulation.Reset();
      simulation.RadialDistribution(&tree_simulation);
      tree_simulation.Write(radial_distribution);
     
      temperature = simulation.ComputeTemperature();
      temperature = temperature / (3.0*K_B);     
      pressure = simulation.ComputePressure();
                
      fprintf(diff, "%f, ", time);
      fflush(diff);
      for (int j = 0; j < diff_tot; j++){	
	if (j < diff_count){	  	 
	  diffusion = simulation.ComputeDiffusion(positions[j]);
	  fprintf(diff, "%f,", diffusion);
	   fflush(diff);
	} else {
	  fprintf(diff, "%f,", 0.0);
	  fflush(diff);
	}
      }
      fprintf(diff, "\n");
      if (info){
	printf("\n Time: %f \n", time);
	printf("--------------\n");
	printf("Temperature: %f \n", temperature);
	printf("Pressure: %f \n", pressure);
	printf("Percent Pruned: %f \n", pct);
      }
      fprintf(stats, "%f %f %f \n", time, pressure,
	      temperature);
      fflush(stats);
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
