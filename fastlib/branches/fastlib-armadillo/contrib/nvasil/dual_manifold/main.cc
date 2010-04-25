/*
 * =====================================================================================
 *
 *       Filename:  main.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/15/2008 10:35:29 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include <errno.h>
#include "fastlib/fastlib.h"
#include "dual_manifold_engine.h"
#include "mvu_dot_prod_objective.h"

void LoadFile(std::string filename, 
    ArrayList<std::pair<index_t, index_t> > &pairs,
    ArrayList<double> &values);

int main(int argc, char* argv[]) {
  fx_init(argc, argv);
  DualManifoldEngine<MVUDotProdObjective> engine;
  std::string train_file=fx_param_str_req(NULL, "train_file"); 
  ArrayList<std::pair<index_t, index_t> > train_pairs_to_consider;
  ArrayList<double> train_dot_prods;
  NOTIFY("Loading data from file %s ...\n", train_file.c_str());
  fx_timer_start(NULL, "data_loading");
  LoadFile(train_file, train_pairs_to_consider, train_dot_prods);
  fx_timer_stop(NULL, "data_loading");
  NOTIFY("Data loaded");
  engine.Init(NULL, train_pairs_to_consider, train_dot_prods);
  NOTIFY("Starting optimization...\n");
  fx_timer_start(NULL, "optimization");
  engine.ComputeLocalOptimum();
  fx_timer_stop(NULL, "optimization");
  NOTIFY("Optimization completed...\n");
  ArrayList<std::pair<index_t, index_t> > test_pairs_to_consider;
  ArrayList<double> test_dot_prods;
  std::string test_file=fx_param_str_req(NULL, "test_file"); 
  NOTIFY("Evaulating test file %s ...\n", test_file.c_str());
  LoadFile(test_file, test_pairs_to_consider, test_dot_prods);
  double test_error=engine.ComputeEvaluationTest(test_pairs_to_consider,test_dot_prods);
  NOTIFY("Test error:%lg ...\n ", test_error);
  std::string result_file = fx_param_str(NULL, "result_file", "results");
  double random_test_error = engine.ComputeRandomEvaluationTest(test_pairs_to_consider,test_dot_prods);
  NOTIFY("Test error for the random predictor:%lg ...\n", random_test_error);
  std::string result_w(result_file);
  std::string result_h(result_file);
  result_w.append("_w.csv");
  result_h.append("_v.csv");
  NOTIFY("Saving results ...\n");
  if (data::Save(result_w.c_str(),*engine.Matrix1())==SUCCESS_FAIL) {
    NONFATAL("Failed to save matrix1 on %s", result_w.c_str());
  }
  if (data::Save(result_h.c_str(),*engine.Matrix2())==SUCCESS_FAIL) {
    NONFATAL("Failed to save matrix2 on %s", result_h.c_str());
  }
  engine.Destruct();
  fx_done();
}

void LoadFile(std::string filename, 
    ArrayList<std::pair<index_t, index_t> > &pairs,
    ArrayList<double> &values) {
  pairs.Init();
  values.Init();
  FILE *fp=fopen(filename.c_str(), "r");
  if (fp==NULL) {
     FATAL("Unable to open file %s, error %s\n", filename.c_str(),
         strerror(errno));
  }
  while (!feof(fp)) {
    index_t user_id;
    index_t movie_id;
    double rating;
    index_t timestamp;
    fscanf(fp, "%i %i %lg %i", &user_id, &movie_id, &rating, &timestamp);
    pairs.PushBackCopy(std::make_pair(user_id-1, 
          movie_id-1));
    values.PushBackCopy(rating);
  }
  fclose(fp);
}


