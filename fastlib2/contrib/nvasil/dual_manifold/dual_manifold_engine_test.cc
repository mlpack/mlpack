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
  void Init() {
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
      fscanf(fp, "%i %i %lg", &user_id, &movie_id, &rating);
      pairs_to_consider.PushBackCopy(std::make_pair(user_id, movie_id));
      dot_prods.PushBackCopy(rating);
    }
    fclose(fp);
    engine_.Init(NULL, pairs_to_consider, dot_prods);
  }
  void Destruct() {
    engine_.Destruct();
  }
  void Test1() {
    NOTIFY("Testing Test1..");
    Init();
    Test1();
    Destruct();
    NOTIFY("Test1 passed!!");
  }
  void TestAll() {
    Test1();
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
