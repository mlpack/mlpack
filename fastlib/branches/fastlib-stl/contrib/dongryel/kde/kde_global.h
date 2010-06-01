#ifndef INSIDE_KDE_PROBLEM_H
#error "This is not a public header file!"
#endif

class MultiTreeGlobal {

 public:

  /** @brief The pointer to the module holding the parameters.
   */
  struct datanode *module;
    
  /** @brief The dimensionality.
   */
  int dimension;

  /** @brief The kernel function.
   */
  TKernelAux kernel_aux;
    
  /** @brief The desired probability level.
   */
  double probability;

  /** @brief The relative error desired.
   */
  double relative_error;

  /** @brief The number of reference points.
   */
  int num_reference_points;

  double normalizing_constant;

  ArrayList<int> hybrid_node_chosen_indices;
  ArrayList<int> query_node_chosen_indices;
  ArrayList<int> reference_node_chosen_indices;
    
  // It is important not to include the module pointer because it will
  // be freed by fx_done()!
  OT_DEF_BASIC(MultiTreeGlobal) {
    OT_MY_OBJECT(dimension);
    OT_MY_OBJECT(kernel_aux);
    OT_MY_OBJECT(probability);
    OT_MY_OBJECT(relative_error);
    OT_MY_OBJECT(num_reference_points);
    OT_MY_OBJECT(normalizing_constant);
    OT_MY_OBJECT(hybrid_node_chosen_indices);
    OT_MY_OBJECT(query_node_chosen_indices);
    OT_MY_OBJECT(reference_node_chosen_indices);
  }
    
 public:

  void Init(int num_queries, int dimension_in, 
	    const ArrayList<Matrix *> &targets, struct datanode *module_in) {
      
    // Set the data node module to incoming one.
    module = module_in;

    hybrid_node_chosen_indices.Init(KdeProblem::num_hybrid_sets);
    query_node_chosen_indices.Init(KdeProblem::num_query_sets);
    reference_node_chosen_indices.Init(KdeProblem::num_reference_sets);
      
    // Set the dimension.
    dimension = dimension_in;

    // Initialize the bandwidth.
    double bandwidth = fx_param_double_req(module, "bandwidth");

    // Initialize the series expansion object.
    if(dimension <= 2) {
      kernel_aux.Init(bandwidth, fx_param_int(module, "order", 7),
		      dimension);
    }
    else if(dimension <= 3) {
      kernel_aux.Init(bandwidth, fx_param_int(module, "order", 5),
		      dimension);
    }
    else if(dimension <= 5) {
      kernel_aux.Init(bandwidth, fx_param_int(module, "order", 3),
		      dimension);
    }
    else if(dimension <= 6) {
      kernel_aux.Init(bandwidth, fx_param_int(module, "order", 1),
		      dimension);
    }
    else {
      kernel_aux.Init(bandwidth, fx_param_int(module, "order", 0),
		      dimension);
    }

    // Set the probability level.
    probability = fx_param_double(module, "probability", 0.9);

    // Set the relative error.
    relative_error = fx_param_double(module, "relative_error", 0.1);
      
    // Set the number of reference points.
    const Matrix &reference_targets = *(targets[0]);
    num_reference_points = reference_targets.n_rows();
      
    // Compute the normalizing constant.
    normalizing_constant = 
      kernel_aux.kernel_.CalcNormConstant(dimension_in) *
      num_reference_points;
      
  }

};
