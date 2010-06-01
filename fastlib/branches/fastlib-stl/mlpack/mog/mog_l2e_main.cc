/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file mog_l2e_main.cc
 * 
 * This program test drives the L2 estimation
 * of a Gaussian Mixture model.
 * 
 * PARAMETERS TO BE INPUT:
 * 
 * --data 
 * This is the file that contains the data on which 
 * the model is to be fit
 *
 * --mog_l2e/K
 * This is the number of gaussians we want to fit
 * on the data, defaults to '1'
 *
 * --output
 * This file will contain the parameters estimated,
 * defaults to 'output.csv'
 *
 */

#include "mog_l2e.h"
#include "mlpack/optimization/optimizers.h"

const fx_entry_doc mog_l2e_main_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   " A file containing the data on which the model"
   " has to be fit.\n"},
  {"output", FX_PARAM, FX_STR, NULL,
   " The file into which the output is to be written into.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc mog_l2e_main_submodules[] = {
  {"mog_l2e", &mog_l2e_doc,
   " Responsible for intializing the model and"
   " computing the parameters.\n"},
   {"opt", &opt_doc,
    " Responsible for minimizing the L2 loss function"
    " and obtaining the parameter values.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc mog_l2e_main_doc = {
  mog_l2e_main_entries, mog_l2e_main_submodules,
  " This program test drives the parametric estimation "
  "of a Gaussian mixture model using L2 loss function.\n"
};

int main(int argc, char* argv[]) {

  fx_module *root = 
    fx_init(argc, argv, &mog_l2e_main_doc);

  ////// READING PARAMETERS AND LOADING DATA //////
  
  const char *data_filename = fx_param_str_req(root, "data");

  Matrix data_points;
  data::Load(data_filename, &data_points);

  ////// MIXTURE OF GAUSSIANS USING L2 ESTIMATION //////

  datanode *mog_l2e_module = fx_submodule(root, "mog_l2e");
  index_t number_of_gaussians = fx_param_int(mog_l2e_module, "K", 1);
  fx_set_param_int(mog_l2e_module, "D", data_points.n_rows());
  index_t dimension = fx_param_int_req(mog_l2e_module, "D");;
  
  ////// RUNNING AN OPTIMIZER TO MINIMIZE THE L2 ERROR //////

  datanode *opt_module = fx_submodule(root, "opt");
  const char *opt_method = fx_param_str(opt_module, "method", "QuasiNewton");
  index_t param_dim = (number_of_gaussians*(dimension+1)*(dimension+2)/2 - 1);
  fx_set_param_int(opt_module, "param_space_dim", param_dim);

  index_t optim_flag = (strcmp(opt_method, "NelderMead") == 0 ? 1 : 0);
  MoGL2E mog;

  if (optim_flag == 1) {

    ////// OPTIMIZER USING NELDER MEAD METHOD //////

    NelderMead opt;
    
    ////// Initializing the optimizer //////
    fx_timer_start(opt_module, "init_opt");
    opt.Init(MoGL2E::L2ErrorForOpt, data_points, opt_module);
    fx_timer_stop(opt_module, "init_opt");

    ////// Getting starting points for the optimization //////
    double **pts;
    pts = (double**)malloc((param_dim+1)*sizeof(double*));
    for(index_t i = 0; i < param_dim+1; i++) {
      pts[i] = (double*)malloc(param_dim*sizeof(double));
    }

    fx_timer_start(opt_module, "get_init_pts");
    MoGL2E::MultiplePointsGenerator(pts, param_dim+1, 
				    data_points, number_of_gaussians);
    fx_timer_stop(opt_module, "get_init_pts");

    ////// The optimization //////
    
    fx_timer_start(opt_module, "optimizing");
    opt.Eval(pts);
    fx_timer_stop(opt_module, "optimizing");
    
    ////// Making model with the optimal parameters //////
    mog.MakeModel(mog_l2e_module, pts[0]);

  }
  else {

    ////// OPTIMIZER USING QUASI NEWTON METHOD //////

    QuasiNewton opt;
    
    ////// Initializing the optimizer //////
    fx_timer_start(opt_module, "init_opt");
    opt.Init(MoGL2E::L2ErrorForOpt, data_points, opt_module);
    fx_timer_stop(opt_module, "init_opt");

    ////// Getting starting point for the optimization //////
    double *pt;
    pt = (double*)malloc(param_dim*sizeof(double));

    fx_timer_start(opt_module, "get_init_pt");
    MoGL2E::InitialPointGenerator(pt, data_points, number_of_gaussians);
    fx_timer_stop(opt_module, "get_init_pt");

    ////// The optimization //////
    
    fx_timer_start(opt_module, "optimizing");
    opt.Eval(pt);
    fx_timer_stop(opt_module, "optimizing");
    
    ////// Making model with optimal parameters //////
    mog.MakeModel(mog_l2e_module, pt);

  }
   
  long double error = mog.L2Error(data_points);
  NOTIFY("Minimum L2 error achieved: %Lf", error);
  mog.Display();
  
  ArrayList<double> results;
  mog.OutputResults(&results);
  
  
  ////// OUTPUT RESULTS //////
  
  const char *output_filename = fx_param_str(NULL, "output", "output.csv");
  
  FILE *output_file = fopen(output_filename, "w");
  
  ot::Print(results, output_file);
  fclose(output_file);
  fx_done(root);

  return 1;
}
