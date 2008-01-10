#include "emst.h"

/* for now, I'm just going to write allnn and make sure I can do it, then try to modify the tree and ordering */

/*template<typename GNP, typename Solver>
void thor::MonochromaticDualTreeMain(datanode *module, const char *gnp_name) {
  const int DATA_CHANNEL = 110;
  const int Q_RESULTS_CHANNEL = 120;
  const int GNP_CHANNEL = 200;
  double results_megs = fx_param_double(module, "results/megs", 1000);
  DistributedCache *points_cache;
  index_t n_points;
  ThorTree<typename GNP::Param, typename GNP::QPoint, typename GNP::QNode> tree;
  DistributedCache q_results;
  typename GNP::Param param;
  
  rpc::Init();
  
  fx_submodule(module, NULL, "io"); // influnce output order
  
  param.Init(fx_submodule(module, gnp_name, gnp_name));
  
  fx_timer_start(module, "read");
  points_cache = new DistributedCache();
  n_points = ReadPoints<typename GNP::QPoint>(
                                              param, DATA_CHANNEL + 0, DATA_CHANNEL + 1,
                                              fx_submodule(module, "data", "data"), points_cache);
  fx_timer_stop(module, "read");
  
  typename GNP::QPoint default_point;
  CacheArray<typename GNP::QPoint>::GetDefaultElement(
                                                      points_cache, &default_point);
  param.SetDimensions(default_point.vec().length(), n_points);
  
  fx_timer_start(module, "tree");
  CreateKdTree<typename GNP::QPoint, typename GNP::QNode>(
                                                          param, DATA_CHANNEL + 2, DATA_CHANNEL + 3,
                                                          fx_submodule(module, "tree", "tree"), n_points, points_cache, &tree);
  fx_timer_stop(module, "tree");
  
  typename GNP::QResult default_result;
  default_result.Init(param);
  tree.CreateResultCache(Q_RESULTS_CHANNEL, default_result,
                         results_megs, &q_results);
  
  typename GNP::GlobalResult global_result;
  RpcDualTree<GNP, Solver>(
                           fx_submodule(module, "gnp", "gnp"), GNP_CHANNEL, param,
                           &tree, &tree, &q_results, &global_result);
  
  rpc::Done();
}
*/


int main(int argc, char* argv[]) {
 
  fx_init(argc, argv);
  
  /*thor::MonochromaticDualTreeMain<Emst, DualTreeDepthFirst<Emst> >(fx_root, "EMST");*/
  
 
  // Include a check parameter here to see if I'm in the multi-processor case
  bool using_thor = fx_param_exists(NULL, "using_thor");
  
  
  if unlikely(using_thor) {
    printf("thor is not yet supported\n");
  } // end if using_thor
  else {
    
    /* Step one: read in the data
    * parameters - leaf size, min width (later: tree type, metric)
    * data - get the number of points and the dimension
    */
    
    
    /* Step two: build the tree
    * TESTING: this section needs to be tested
    */
    
    fx_timer_start(NULL, "tree_building");
    
        
    Emst_Tree* this_tree;
    Matrix data_matrix;
    ArrayList<index_t> data_permutation;
    
    //printf("Building tree\n");
    struct datanode* tree_params = fx_param_node(NULL, "tree_params");
    int leaflen = fx_param_int(tree_params, "./leaflen", 1);
    tree::LoadKdTree(fx_submodule(NULL, "tree_params", "data"), &data_matrix, &this_tree, &data_permutation);
    
    
    fx_timer_stop(NULL, "tree_building");
    
    /* Step three: run the algorithm 
    * Find all nearest neighbors
    * Add the pairs to the edge list
    * Update the component information and run tree cleanup
    */
    
        
    DualTreeBoruvka dtb;
    dtb.Init(this_tree);
    
    //dtb.TestTree();
    
    dtb.ComputeMST();
    
    /* Step four: format and output results
    */
    
    //dtb.output_results();
    
  }// end else (if using_thor)
  
  fx_done();
  
}