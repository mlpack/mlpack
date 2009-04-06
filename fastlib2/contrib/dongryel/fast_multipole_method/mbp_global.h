/** @brief Defines the global variable for the Axilrod-Teller force
 *         computation.
 */
class MultiTreeGlobal {

 public:
  
  /** @brief The module holding the parameters.
   */
  struct datanode *module;

  /** @brief The kernel object.
   */
  TKernel kernel_aux;

  /** @brief The chosen indices.
   */
  ArrayList<index_t> hybrid_node_chosen_indices;

  ArrayList<index_t> query_node_chosen_indices;
    
  ArrayList<index_t> reference_node_chosen_indices;

  /** @brief The total number of 3-tuples that contain a particular
   *         particle.
   */
  double total_n_minus_one_tuples;
    
  /** @brief The relative error requirement.
   */
  double relative_error;

  /** @brief The probability requirement.
   */
  double probability;

  /** @brief The standard score that corresponds to the probability
   *         requirement.
   */
  double z_score;

  int dimension;

 public:

  void Init(index_t total_num_particles, index_t dimension_in,
	    const ArrayList<Matrix *> &reference_targets,
	    struct datanode *module_in) {

    hybrid_node_chosen_indices.Init(TKernel::order);
      
    total_n_minus_one_tuples = 
      math::BinomialCoefficient(total_num_particles - 1,
				TKernel::order - 1);

    // Set the incoming module for referring to parameters.
    module = module_in;

    // Extract the relative error/probability and the bandwidth from
    // the module.
    relative_error = fx_param_double(module, "relative_error", 0.1);
    probability = fx_param_double(module, "probability", 1.0);
    z_score = InverseNormalCDF::Compute(probability + 0.5 * (1 - probability));
    kernel_aux.Init(fx_param_double(module, "bandwidth", 0.3));

    // Set the dimension.
    dimension = dimension_in;
  }

};
