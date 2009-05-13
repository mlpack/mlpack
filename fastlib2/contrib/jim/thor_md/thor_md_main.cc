#include "fastlib/fastlib_int.h"
#include "fastlib/thor/thor.h"
#include "thor_md.h"

const fx_entry_doc root_entries[] = { 
  {"dt", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies time step of dynamic simulation. \n"},
  {"tf", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies duration of simulation \n"},
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
  ThorMD simulation;

  // Output files
  const char* fp_coords;
  FILE *coords;
  fp_coords = fx_param_str(NULL, "coord", "coords.dat");

  double time = 0, time_step_, final_time_;
  double set_temp_, temperature_;
  // Get Input values
  time_step_ = fx_param_double(0, "dt", 0.1);
  final_time_ = fx_param_double(0, "tf", 1.0e3);
  set_temp_ = fx_param_double(0, "temp", -1.0);
  

  // Read in positions and velocities
  struct datanode* parameters = fx_submodule(root, "param");
  simulation.Init(parameters);
  simulation.Compute(parameters);

  // Open files for output


  // Loop over all time steps
  while (time < final_time_){

    // Update positions
    simulation.UpdatePoints(time_step_);
    simulation.RebuildTree(parameters);
    simulation.Compute(parameters);
    // Take snapshot, if nesc.
    if (0){
    }

    // Do we get stats this time?
    if ((int)(time / time_step_ -0.5) % 5 == 0){      
      // Get radial distribution
      // Get diffusion, temp, etc.
      temperature_ = simulation.global_result_.old_temp_;
      temperature_ = temperature_ / K_B;
      printf("Temperature: %f \n", temperature_);
      if (set_temp_ > 0){
	double ratio = sqrt(set_temp_ / temperature_);
	simulation.ScaleToTemperature(ratio);
      }
    }    
       

    time = time + time_step_;
  }

  Matrix positions;
  simulation.GetFinalPositions(&positions);
  coords = fopen(fp_coords, "w+");
  for (int i = 0; i < positions.n_cols(); i++){
    for (int j = 0; j < positions.n_rows()-1; j++){
      fprintf(coords, "%f,\t ", positions.get(j,i));
    }
    fprintf(coords, "%f\n", positions.get(positions.n_rows()-1, i));
  }

  // Close output files
  simulation.Fin();

  //fx_done(fx_root);
  return 0;
}
