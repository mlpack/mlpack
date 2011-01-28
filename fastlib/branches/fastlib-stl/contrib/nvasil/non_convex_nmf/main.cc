#include "fastlib/fastlib.h"
#include <string>
#include <algorithm>
#include "mlpack/allknn/allknn.h"
#include "nmf_engine.h"
#include "../mvu/mvu_objectives.h"

 //typedef NmfEngine<ClassicNmfObjective> Engine;
 //typedef NmfEngine<BigSdpNmfObjectiveMaxVarIsometric> Engine;
 //typedef NmfEngine<BigSdpNmfObjectiveMaxFurthestIsometric> Engine;
 //typedef NmfEngine<NmfObjectiveIsometric> Engine;
template<typename Engine>
void Execute(fx_module *module); 
void ComputeSpectrum(Matrix &w_mat, Matrix &h_mat, Matrix *spectrum);
void ComputeNeighborhoodErrors(ArrayList<std::pair<index_t, index_t> > &neighbor_pairs, 
                               ArrayList<double> &distances,
                               Matrix &new_points,
                               index_t knns,
                               double *error);


int main(int argc, char *argv[]) {
  fx_module *fx_root;
	fx_root=fx_init(argc, argv, NULL);
  std::string nmf_type=fx_param_str(fx_root, "/engine/nmf_type", "classic");
	if (nmf_type==std::string("classic")) {
    Execute<NmfEngine<ClassicNmfObjective> >(fx_root);
  } else {
    if (nmf_type==std::string("isometric")) {
      Execute<NmfEngine<BigSdpNmfObjectiveMaxFurthestIsometric> >(fx_root);
    } else {
      FATAL("This nmf method %s is not supported\n", nmf_type.c_str());
    }
  }
  
  fx_done(fx_root);
  return 0;
}

template<typename Engine>
void Execute(fx_module *module) {
  fx_module *nmf_module=fx_submodule(module, "/engine");
  index_t num_of_restarts=fx_param_int(module, "/engine/num_of_restarts",1);
  Engine *engine;
/*	fx_set_param_int(nmf_module, "sdp_rank", 3);
  fx_set_param_int(nmf_module, "new_dimension", 2);
  fx_set_param_int(nmf_module, "optfun/knns", 7);
  fx_set_param_double(nmf_module, "optfun/grad_tolerance", 1e-6);
  fx_set_param_double(nmf_module, "optfun/feasibility_error",40);
	fx_set_param_double(nmf_module, "l_bfgs/sigma", 10);
	fx_set_param_double(nmf_module, "l_bfgs/gamma", 5);
 	fx_set_param_double(nmf_module, "l_bfgs/step_size", 3);
  fx_set_param_int(nmf_module, "l_bfgs/mem_bfgs", 5);
	fx_set_param_bool(nmf_module, "l_bfgs/silent", false);
  fx_set_param_bool(nmf_module, "l_bfgs/use_default_termination", false);
*/

  AllkNN allknn;
  Matrix data_mat;
  data::Load(fx_param_str_req(module, "/engine/data_file"), &data_mat);
  index_t knns=fx_param_int(module, "/engine/knns", 3);
  allknn.Init(data_mat, fx_submodule(module, "/engine"));
  ArrayList<index_t> neighbors;
  ArrayList<double> distances;
  allknn.ComputeNeighbors(&neighbors,
                          &distances);
  ArrayList<std::pair<index_t, index_t> > neighbor_pairs;
  ArrayList<double> neighbor_distances;
  index_t num_of_pairs;
  MaxVarianceUtils::ConsolidateNeighbors(neighbors,
      distances,
      knns,
      knns,
      &neighbor_pairs,
      &neighbor_distances,
      &num_of_pairs);


  double resulting_score[num_of_restarts];
  double resulting_neighborhood_error[num_of_restarts]; 
  for(index_t i=0; i<num_of_restarts; i++) {
    engine = new Engine();
    engine->Init(nmf_module);
    engine->ComputeNmf();
    resulting_score[i]=engine->reconstruction_error();
	  Matrix w_mat;
	  Matrix h_mat;
    engine->GetW(&w_mat);
	  engine->GetH(&h_mat);
    char matrix_filename[128];
    sprintf(matrix_filename, "W_iteration_%i.csv", i);
    data::Save(matrix_filename, w_mat);
    sprintf(matrix_filename, "H_iteration_%i.csv", i);
	  data::Save(matrix_filename, h_mat);
    Matrix spectrum;
    ComputeSpectrum(w_mat, h_mat, &spectrum);
    sprintf(matrix_filename, "spectrum_iteration_%i.csv", i);
    data::Save(matrix_filename, spectrum);
    ComputeNeighborhoodErrors(neighbor_pairs, neighbor_distances, w_mat, knns, 
                              resulting_neighborhood_error+i);
    
    delete engine;
  }
  fx_result_double_array(module, "percentage_reconstruction_error",
			    num_of_restarts, resulting_score);
  fx_result_double_array(module, "perencentage_neighborhood_error", 
          num_of_restarts, resulting_neighborhood_error);
  double best_error=*std::min_element(resulting_score, resulting_score+num_of_restarts);
  fx_result_double(module, "best_error", best_error);

}

void ComputeSpectrum(Matrix &w_mat, Matrix &h_mat, Matrix *spectrum) {
  spectrum->Init(w_mat.n_rows(), 1);
  spectrum->SetAll(0.0);
  double total_sum=0;
  for(index_t i=0; i<w_mat.n_rows(); i++) {
    for(index_t j=0; j<w_mat.n_cols(); j++) {
      spectrum->set(i, 0, spectrum->get(i, 0)+std::pow(w_mat.get(i, j),2));
    }
  }
  Matrix h_norms;
  h_norms.Init(h_mat.n_rows(), 1);
  h_norms.SetAll(0.0);
  for(index_t i=0; i<h_mat.n_rows(); i++) {
    for(index_t j=0; j<h_mat.n_cols(); j++) {
      h_norms.set(i, 0, h_norms.get(i, 0) + math::Sqr(h_mat.get(i, j)));
    }
  }
  for(index_t i=0; i<h_norms.n_rows(); i++) {
    spectrum->set(i, 0, spectrum->get(i, 0)/std::sqrt(h_norms.get(i,0)));
  }
  std::sort(spectrum->ptr(), 
            spectrum->ptr()+spectrum->n_rows(), 
            std::greater<double>());
}

void ComputeNeighborhoodErrors(ArrayList<std::pair<index_t, index_t> > &neighbor_pairs, 
                               ArrayList<double> &distances,
                               Matrix &new_points,
                               index_t knns,
                               double *error) {
  *error=0;
  double total_distances=0;
  for(index_t i=0; i<neighbor_pairs.size(); i++) {
    index_t n1=neighbor_pairs[i].first;
    index_t n2=neighbor_pairs[i].second;
    *error += std::pow(la::DistanceSqEuclidean(new_points.n_rows(), 
                                     new_points.GetColumnPtr(n1),
                                     new_points.GetColumnPtr(n2))
                     -distances[i],2);
      total_distances += distances[i]*distances[i];
  }
  *error=100.0 * std::sqrt(*error/total_distances);
}
  
