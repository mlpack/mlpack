/*
 * =====================================================================================
 *
 *       Filename:  dual_manifold_engine_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/11/2008 11:21:53 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include <string>
#include "fastlib/fastlib.h"
#include "dual_manifold_engine.h"
#include "mvu_dot_prod_objective.h"
#include "errno.h"

class DualManifoldEngineTest {
 public:
  void Test1() {
    NOTIFY("Testing Test1..");
    std::string filename= "D_mat.txt"; 
    FILE *fp=fopen(filename.c_str(), "r");
    if (fp==NULL) {
     FATAL("Unable to open file %s, error %s\n", filename.c_str(),
         strerror(errno));
    }
    ArrayList<std::pair<index_t, index_t> > pairs_to_consider;
    pairs_to_consider.Init();
    ArrayList<double> dot_prods;
    dot_prods.Init();
    while (!feof(fp)) {
      double id1;
      double id2;
      double rating;
      fscanf(fp, "%lg %lg %lg", &id1, &id2, &rating);
      pairs_to_consider.PushBackCopy(std::make_pair((index_t)id1-1, 
            (index_t)id2-1));
      dot_prods.PushBackCopy(rating);
    }
    fclose(fp);
    engine_.Init(NULL, pairs_to_consider, dot_prods);
    engine_.ComputeLocalOptimum();
    NOTIFY("Test1 passed!!");
  }
  
  void Test2() {
    NOTIFY("Testing Test2..");
    datanode *node=fx_submodule(NULL, "", "opts");
    char buffer[128];
    sprintf(buffer, "%i", 40);
    fx_set_param(node, "components", buffer);

    std::string filename="/net/hg200/nvasil/dataset/ml-data_0/u.data";
    FILE *fp=fopen(filename.c_str(), "r");
    if (fp==NULL) {
     FATAL("Unable to open file %s, error %s\n", filename.c_str(),
         strerror(errno));
    }
    ArrayList<std::pair<index_t, index_t> > pairs_to_consider;
    pairs_to_consider.Init();
    ArrayList<double> dot_prods;
    dot_prods.Init();
    while (!feof(fp)) {
      index_t user_id;
      index_t movie_id;
      double rating;
      index_t timestamp;
      fscanf(fp, "%i %i %lg %i", &user_id, &movie_id, &rating, &timestamp);
      pairs_to_consider.PushBackCopy(std::make_pair(user_id-1, 
            movie_id-1));
      dot_prods.PushBackCopy(rating);
    }
    fclose(fp);
    engine_.Init(NULL, pairs_to_consider, dot_prods);
    engine_.ComputeLocalOptimum();
    NOTIFY("Test2 passed!!");
  }

  void Destruct() {
    engine_.Destruct();
  }
 
  void TestAll() {
    Test1();
    Destruct();
    Test2();
    Destruct();
  }
 
 private:
  DualManifoldEngine<MVUDotProdObjective> engine_;
};

int main(int argc, char * argv[]) {
  fx_init(argc, argv);
  DualManifoldEngineTest test;
  test.TestAll();
  fx_done();
}
