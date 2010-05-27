#include "fastlib/fastlib_int.h"
#include "fastlib/thor/thor.h"
#include "thor_md.h"

const fx_entry_doc root_entries[] = { 
  {"dt", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies time step of dynamic simulation. \n"},
  {"tf", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies duration of simulation \n"},
  {"snapshots", FX_PARAM, FX_DOUBLE, NULL,
   "Number of starting points for diffusion calculation"},
  {"diff", FX_PARAM, FX_STR, NULL,
   "Output file for diffusion data"},
  {"temp", FX_PARAM, FX_DOUBLE, NULL,
   "Set temperature. If no temp is specified, simulation runs at const. energy."},
  FX_ENTRY_DOC_DONE  
};

const fx_submodule_doc md_submodules[] = {
   {"param", &param_doc, "Parameters for MD-simulation \n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc root_doc = {
  root_entries,  md_submodules,
  "Simulation Parameters \n"
};

static const double K_B = 8.61734e-5;

int main(int argc, char *argv[]) {  
  fx_module *root = fx_init(argc, argv, &root_doc);
  rpc::Init();
  ThorMD simulation;

  // Output files
  const char* fp_coords;
  const char* fp_diff;
  const char* fp_stats;

  FILE *coords;
  FILE *diff;
  FILE *stats;

  fp_coords = fx_param_str(NULL, "coord", "coords.dat");  
 
  double time = 0, time_step_, final_time_;
  double set_temp_, temperature_, pressure_;
  // Get Input values
  time_step_ = fx_param_double(0, "dt", 0.1);
  final_time_ = fx_param_double(0, "tf", 1.0e3);
  set_temp_ = fx_param_double(0, "temp", -1.0);
  

  // Read in positions and velocities
  struct datanode* parameters = fx_submodule(root, "param");
  
  simulation.Init(parameters);
  //  simulation.Compute(parameters);

  // Open files for output

  // Diffusion computation stuff
  int diff_tot, diff_count = 0;
  double delta = 100*time_step_, last_time = -2*delta;  
  ArrayList<Matrix> reference_positions;    
  if (rpc::is_root()){
    fp_diff = fx_param_str(NULL, "diff", "diffusion.dat");
    diff = fopen(fp_diff, "w+");  
    fp_stats = fx_param_str(NULL, "stats", "stats.dat");
    stats = fopen(fp_stats, "w+");  
  }
  diff_tot = fx_param_int(0, "snapshots", 1);   
  reference_positions.Init(diff_tot);
     
  
  // Loop over all time steps
  while (time < final_time_){

    // Update positions
    simulation.Compute(parameters);     
    simulation.UpdatePoints(time_step_);  
    simulation.RebuildTree(parameters);
    // Take snapshot, if nesc.      
    if (diff_count < diff_tot && time > last_time + delta){
      last_time = time;    
      simulation.TakeSnapshot(&reference_positions[diff_count]);
      diff_count++;    
    }
    
    // Do we get stats this time?
    if (((int)(time / time_step_ -0.5) % 5 == 0)){      
      // Get radial distribution
      // Get diffusion, temp, etc.
      if (rpc::is_root()){
	fprintf(diff, "%f, ", time);
      }
      for (int j = 0; j < diff_tot; j++){	
	double diffusion;
	if (j < diff_count){	  	 
	  diffusion = simulation.GetDiffusion(reference_positions[j]);
	  if (rpc::is_root()){
	    fprintf(diff, "%f,", diffusion);	  
	  }
	} else {
	  if (rpc::is_root()){
	    fprintf(diff, "%f,", 0.0);	 
	  }
	}
      }   
    
      if (rpc::is_root()){
	fprintf(diff, "\n");
	temperature_ = simulation.global_result_.old_temp_;
	temperature_ = temperature_ / K_B;
	pressure_ = simulation.global_result_.pressure_;
	printf("Time: %f \n--------------\n", time);
	printf("Temperature: %f \n", temperature_);
	printf("Pressure: %f \n \n", pressure_);	
	fprintf(stats, "% f %f %f \n", time, temperature_, pressure_);
      }     
      if (set_temp_ > 0){
	double ratio = sqrt(set_temp_ / temperature_);
	simulation.ScaleToTemperature(ratio);
      }
    }      
    time = time + time_step_;
  }

  
 
  Matrix positions;
  simulation.GetFinalPositions(&positions);
  if (rpc::is_root()){
    coords = fopen(fp_coords, "w+");
    for (int i = 0; i < positions.n_cols(); i++){
      for (int j = 0; j < positions.n_rows()-1; j++){
	fprintf(coords, "%f,\t ", positions.get(j,i));
      }
      fprintf(coords, "%f\n", positions.get(positions.n_rows()-1, i));
    }
  }
  

  // Close output files
  simulation.Fin();
  if (rpc::is_root()){
    fclose(diff);
    fclose(stats);
  }
  
  fx_done(fx_root);
  return 0;
}
