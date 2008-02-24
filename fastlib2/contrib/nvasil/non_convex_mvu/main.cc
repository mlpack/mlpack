#include "fastlib/fastlib.h"
#include "non_convex_mvu.h"
#include <string>
#include <algorithm>
/**
 * This file will run non convex mvu with differnt settings
 * 
 */
void SetParameters(NonConvexMVU &engine);
void LoadData(std::string data_file, 
              double sample_factor, 
              Matrix *data); 

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  NonConvexMVU engine;
  Matrix data;
  std::string data_file=fx_param_str_req(NULL, "data_file");
  double sample_factor=fx_param_double(NULL, "sample_factor", 1.0);
  LoadData(data_file, sample_factor, &data);
  
  std::string opt_method=fx_param_str(NULL, "opt_method", "bfgs_max_var");
  SetParameters(engine);
  fx_timer_start(NULL, "tree_init");
  if (opt_method=="spe") {
    engine.Init<StochasticGrad>(data); 
  } else {
    engine.Init<DeterministicGrad>(data); 
  }
  fx_timer_stop(NULL, "tree_init");
  // Optimization method to be used
  if (opt_method=="bfgs_max_var"){
    fx_timer_start(NULL, "optimization");
    engine.ComputeLocalOptimumBFGS<MaxVariance, 
                                    EqualityOnNearest,
                                    DeterministicGrad>();

    fx_timer_stop(NULL,  "optimization"); 
  } else {
    if (opt_method=="bfgs_max_furth") {
      fx_timer_start(NULL, "optimization");
      engine.ComputeLocalOptimumBFGS<MaxFurthestNeighbors, 
                                      EqualityOnNearest,
                                      DeterministicGrad>();
      fx_timer_stop(NULL,  "optimization"); 
 
    } else {
      if (opt_method=="spe") {
        fx_timer_start(NULL, "optimization");
        engine.ComputeLocalOptimumBFGS<MaxFurthestNeighbors, 
                                      EqualityOnNearest,
                                      DeterministicGrad>();
        fx_timer_stop(NULL,  "optimization"); 
      } else {
        FATAL("You didn't select a valid optimization method \n");
      }
    }
  }
      
  std::string out_file = fx_param_str(NULL, "out_file", "results.csv");
  data::Save(out_file.c_str(), engine.coordinates());
  fx_done();
}

void SetParameters(NonConvexMVU &engine) {
  index_t knns = fx_param_int(NULL, "knns", 4);
  engine.set_knns(knns);
  index_t kfns = fx_param_int(NULL, "kfns", 0);
  engine.set_kfns(kfns);
  index_t leaf_size=fx_param_int(NULL, "leaf_size", 20);
  engine.set_leaf_size(leaf_size);
  double eta=fx_param_double(NULL, "eta", 0.99);
  engine.set_eta(eta);
  double gamma=fx_param_double(NULL, "gamma", 2);
  engine.set_gamma(gamma);
  double sigma=fx_param_double(NULL, "sigma", 1);
  engine.set_sigma(sigma);
  double step_size=fx_param_double(NULL, "step_size", 1);
  engine.set_step_size(step_size);
  index_t max_iterations=fx_param_int(NULL, "max_iterations", 10000);
  engine.set_max_iterations(max_iterations);
  index_t new_dimension=fx_param_int(NULL, "new_dimension", 3);
  engine.set_new_dimension(new_dimension);
  double dtolerance=fx_param_double(NULL, "dtolerance", 1e-3);
  engine.set_distance_tolerance(dtolerance); 
  double wolfe_sigma1 = fx_param_double(NULL, "wolfe_sigma1", 0.1);
  double wolfe_sigma2 = fx_param_double(NULL, "wolfe_sigma2", 0.9);
  engine.set_wolfe_sigma(wolfe_sigma1, wolfe_sigma2);
  double wolfe_beta = fx_param_double(NULL, "wolfe_beta", 0.8);
  engine.set_wolfe_beta(wolfe_beta);
  index_t mem_bfgs = fx_param_int(NULL, "mem_bfgs", 50);
  engine.set_mem_bfgs(mem_bfgs);
}

void LoadData(std::string data_file, 
              double sample_factor, 
              Matrix *data) {
  if unlikely(sample_factor>1 or sample_factor<0) {
    FATAL("Sample factor cannot be greater than 1 and less than zero\n");
  }
  if (sample_factor==1.0) {
    data::Load(data_file.c_str(), data);
  } else {
    Matrix temp;
    data::Load(data_file.c_str(), &temp);
    index_t num_of_points = temp.n_cols();
    index_t reduced_num_of_points = index_t(num_of_points *sample_factor);
    index_t *permutations = new index_t[num_of_points];
    for(index_t i=0; i<num_of_points; i++) {
      permutations[i]=i;
    }
    std::random_shuffle(permutations, permutations+num_of_points);
    data->Init(temp.n_rows(), reduced_num_of_points);
    for(index_t i=0; i<reduced_num_of_points; i++) {
      memcpy(data->GetColumnPtr(i), 
             temp.GetColumnPtr(permutations[i]), 
             sizeof(double)*temp.n_rows());
    }
    delete []permutations;
  }
}
