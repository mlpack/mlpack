/**
 * @file dual_tree_integrals.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Code for computing two electron integrals with a dual-tree approximation
 */

#ifndef DUAL_TREE_INTEGRALS_H
#define DUAL_TREE_INTEGRALS_H

#include <fastlib/fastlib.h>

/**
 * Algorithm class for computing two-electron integrals.
 *
 * This object should still exist between SCF iterations, rather than being 
 * reconstructed each time.
 */
class DualTreeIntegrals {
  
  FORBID_ACCIDENTAL_COPIES(DualTreeIntegrals);
  
  friend class DualTreeIntegralsTest;
  
 public:
  DualTreeIntegrals() {}
  
  ~DualTreeIntegrals() {}
  
 private:
  /**
   * Stat class for tree building.  
   *
   * Maybe something for allocating error.  
   */
  class IntegralStat {
   
   private:
   
    // The node's index in a pre-order depth-first traversal of the tree
    index_t node_index_;
    
    // I might need some notion of bandwidths
    double min_bandwidth_;
    double max_bandwidth_;
    
    index_t height_;
    
    double density_upper_bound_;
    double density_lower_bound_;
    
   public:
    void Init() {
      
      node_index_ = -1;
      min_bandwidth_ = 0.0;
      max_bandwidth_ = DBL_MAX;
      
    } // Init
    
    void Init(const Matrix& matrix, index_t start, index_t count) {
      
      Init();
      height_ = 0;
      
    } // Init (leaves)
    
    void Init(const Matrix& matrix, index_t start, index_t count, 
              const IntegralStat& left, const IntegralStat& right) {
    
      Init();
      height_ = max(left.height(), right.height()) + 1;
      
    } // Init (non-leaves)
    
    index_t height() const {
      return height_;
    } // height()
    
    void set_height(index_t new_height) {
      height_ = new_height;
    } // set_height
    
    index_t node_index() const {
      return node_index_;
    } // node_index
    
    void set_node_index(index_t new_index) {
      node_index_ =  new_index;
    } // set_node_index
    
    double min_bandwidth() const {
      return min_bandwidth_; 
    } // min_bandwdith
    
    void set_min_bandwidth(double new_min) {
      min_bandwidth_ = new_min;
    } //set_min_bandwidth
    
    void set_max_bandwidth(double new_max) {
      max_bandwidth_ = new_max;
    } // set_max_bandwidth
    
    double max_bandwidth() const {
      return max_bandwidth_;
    } // max_bandwidth_
    
    void set_density_upper_bound(double upper_bound) {
      density_upper_bound_ = upper_bound;
    } // set_density_upper_bound_()
    
    double density_upper_bound() const {
      return density_upper_bound_;
    } // density_upper_bound()

    void set_density_lower_bound(double lower_bound) {
        density_lower_bound_ = lower_bound;
      } // set_density_lower_bound_()
    
    double density_lower_bound() const {
      return density_lower_bound_;
    } // density_lower_bound()
      
    
  }; // class IntegralStat
  
 public:
  // This assumes identical bandwidth small Gaussians
  // Otherwise, I'll need something other than a Matrix
  // I should also consider something better than bounding boxes
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, IntegralStat> IntegralTree; 
  
 private:
    
  // The tree 
  IntegralTree* tree_;
  
  // The centers of the identical width, spherical Gaussian basis functions
  Matrix centers_;
  
  // The fx module
  struct datanode* module_;
  
  // The common bandwidth of all the basis functions
  double bandwidth_;
  
  // The number of times an approximation is invoked
  int number_of_approximations_;
  
  // The number of times the base case is called
  int number_of_base_cases_;
  
  // The value eps where the returned integrals are within 1-eps of the true 
  // value
  double epsilon_;
  
  // The return values are stored here
  // total_integrals_.ref(i, j) is the fock matrix entry i, j
  Matrix fock_matrix_;
  
  // The exchange contribution
  Matrix exchange_matrix_;
  
  // The coulomb contribution
  Matrix coulomb_matrix_;
  
  // The density matrix, will be input for now
  Matrix density_matrix_;
  
  // The total number of basis functions
  index_t number_of_basis_functions_;
  
  index_t traversal_index_;
  
  // Stores the permutation used in tree-building
  ArrayList<index_t> old_from_new_centers_;
  
  // Size of leaves in the tree
  int leaf_size_;
  
  // The normalization constant to the fourth power
  double normalization_constant_fourth_;
  

  /////////////////////// Functions ///////////////////////////////////
  
  /**
   * Does a preorder traversal of the tree to set remaining statistics in the
   * nodes.  
   *
   * TODO: this can be used to compute density matrix bounds when I include them
   */
  void PreOrderTraversal_(IntegralTree* this_node) {
    
    if (this_node->is_leaf()) {
     
      this_node->stat().set_node_index(traversal_index_);
      
    }
    else {
      
      PreOrderTraversal_(this_node->left());
      PreOrderTraversal_(this_node->right());
      
      this_node->stat().set_node_index(traversal_index_);
         
    }
   
    traversal_index_ = traversal_index_ + 1;
    
  } // PreOrderTraversal  
  
    
  /**
   * Determines if the integral among the four nodes can be approximated
   *
   * May need some idea of the amount of error allowed for these nodes
   */
  bool CanApproximate_(IntegralTree* mu, IntegralTree* nu, IntegralTree* rho, 
                 IntegralTree* sigma, double* approximate_value) { 
    
    bool can_prune = false;
    
    double rho_sigma_min_dist = rho->bound().MinDistanceSq(sigma->bound());
    double rho_sigma_max_dist = rho->bound().MaxDistanceSq(sigma->bound());
    //double rho_sigma_mid_dist = rho->bound().MidDistanceSq(sigma->bound());
    
    double mu_nu_min_dist = mu->bound().MinDistanceSq(nu->bound());
    double mu_nu_max_dist = mu->bound().MaxDistanceSq(nu->bound());
    //double mu_nu_mid_dist = mu->bound().MidDistanceSq(nu->bound());
    
    double mu_rho_min_dist = mu->bound().MinDistanceSq(rho->bound());
    double mu_rho_max_dist = mu->bound().MaxDistanceSq(rho->bound());
    //double mu_rho_mid_dist = mu->bound().MidDistanceSq(rho->bound());
    
    double nu_sigma_min_dist = nu->bound().MinDistanceSq(sigma->bound());
    double nu_sigma_max_dist = nu->bound().MaxDistanceSq(sigma->bound());
    //double nu_sigma_mid_dist = nu->bound().MidDistanceSq(sigma->bound());
    
    double up_bound = ComputeSingleIntegral_(mu_nu_min_dist, rho_sigma_min_dist, 
        mu_rho_min_dist, nu_sigma_min_dist);
    double low_bound = ComputeSingleIntegral_(mu_nu_max_dist, 
        rho_sigma_max_dist, mu_rho_max_dist, nu_sigma_max_dist);
    
    // This only matters in the absolute error case
    up_bound = up_bound * normalization_constant_fourth_;
    low_bound = low_bound * normalization_constant_fourth_;
    
    //printf("up_bound = %g, low_bound = %g\n", up_bound, low_bound);
    
    DEBUG_ASSERT(up_bound >= low_bound);
    
    // I think I need to account for symmetry here
    // Multiply by 2 because each reference pair counts twice ?
    double my_allowed_error = epsilon_ * rho->count() * sigma->count()
        / (number_of_basis_functions_ * number_of_basis_functions_);
    
    // The total error I'm incurring is the max error for one integral times 
    // the number of approximations I'm making
    double my_max_error = 0.5 * (up_bound - low_bound) * rho->count() * 
        sigma->count();
    
        
    if (my_max_error < my_allowed_error) {
      
      //DEBUG_ASSERT(my_max_error < epsilon_);
      
      can_prune = true;
      
      // Maybe account for symmetry here too
      *approximate_value = 0.5 * (up_bound + low_bound);
      //printf("approx_val = %g\n", *approximate_value);
      
    }
    else {
      DEBUG_ONLY(*approximate_value = BIG_BAD_NUMBER);
    }
    
    return can_prune;
    
  } // CanApproximate_
  
  /**
   * Computes the function F from my notes, which is similar to erf 
   *
   * TODO: Consider inlining this function
   *
   * Also, the integral project notes have a slightly different definition of
   * this function.  I should make sure they're compatible.  
   */
  double ErfLikeFunction_(double z) {
    
    if (z == 0) {
      return 1.0;
    }
    else {
      return((1/sqrt(z)) * sqrt(math::PI) * 0.5 * erf(sqrt(z)));
    }
  
  } // ErfLikeFunction_
  
  /**
   * Computes a single integral based on the distances between the centers.  
   * Intended for use in bounding.  
   */
  double ComputeSingleIntegral_(double mu_nu_dist, double rho_sigma_dist, 
                                     double mu_rho_dist, double nu_sigma_dist) {
    
    double return_value;
    
    return_value = 0.25 * pow(math::PI, 2.5);
    
    // the 0.25 comes from the four center distance identity
    return_value = return_value * ErfLikeFunction_(bandwidth_ * 0.25 * 
                                                 (mu_rho_dist + nu_sigma_dist));
    
    // I added a factor of 1/2, another mistake I think
    return_value = return_value * 
        exp(-0.5 * bandwidth_ * (mu_nu_dist + rho_sigma_dist));
    
    return return_value;
    
  } // ComputeSingleIntegral_
  
  /**
   * Finds the single integral of four Gaussians at the given centers
   *
   *
   * Am I including the density matrix in these computations?
   *
   * TODO: consider rewriting this to take the distances as the arguments so 
   * that it can be used for pruning computations as well.  
   *
   * Also, rewrite the math to take advantage of the mixed centers identity
   */
  double ComputeSingleIntegral_(const Vector& mu_center, 
                                const Vector& nu_center, 
                                const Vector& rho_center, 
                                const Vector& sigma_center) {
    
    // Should just be able to plug in the formula
    
    // Constant in front
    double return_value = 0.25 * pow((math::PI/bandwidth_), 2.5);
    
    double four_centers_dists = la::DistanceSqEuclidean(mu_center, rho_center) + 
        la::DistanceSqEuclidean(nu_center, sigma_center);
            
    // F(\alpha d^2(x_{i j} x_{k l}))
    // equivalent to F(\alpha 
    return_value = return_value * 
        ErfLikeFunction_(bandwidth_ * 0.25 * four_centers_dists);
    
    
    // exp(-\alpha (d^2(x_i, x_j) + d^2(x_k, x_l))
    double between_centers_dists = 
        la::DistanceSqEuclidean(mu_center, nu_center) 
        + la::DistanceSqEuclidean(rho_center, sigma_center);
    
    return_value = return_value * 
        exp(-0.5 * bandwidth_ * between_centers_dists);
        
    //printf("computing integral: %g\n", return_value);
        
    return return_value;
    
  } // ComputeSingleIntegral_
  
  /**
   * Exhaustively computes the total integrals among the four nodes 
   *
   * TODO: account for symmetry here, particularly the mu+nu/rho+sigma kind
   *
   * Bug: Does it overcount when the nodes have more than one point?  I think 
   * it does in the case where rho and sigma are the same node.  There is a 
   * similar problem when mu and nu are the same.
   */
  void ComputeIntegralsBaseCase_(IntegralTree* mu, IntegralTree* nu, 
                                 IntegralTree* rho, IntegralTree* sigma) {
    
    for (index_t mu_index = mu->begin(); mu_index < mu->end(); mu_index++) {
     
      for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
       
        double integral_value = total_integrals_.ref(mu_index, nu_index);
        
        for (index_t rho_index = rho->begin(); rho_index < rho->end();
             rho_index++) {
         
          for (index_t sigma_index = sigma->begin(); sigma_index < sigma->end(); 
               sigma_index++) {
            
            
            Vector mu_vec;
            centers_.MakeColumnVector(mu_index, &mu_vec);
            Vector nu_vec;
            centers_.MakeColumnVector(nu_index, &nu_vec);
            Vector rho_vec;
            centers_.MakeColumnVector(rho_index, &rho_vec);
            Vector sigma_vec;
            centers_.MakeColumnVector(sigma_index, &sigma_vec);
            
            
            // Multiply by normalization to the fourth, since it appears 
            // once in each of the four integrals
            
            double this_integral = density_matrix_.ref(rho_index, sigma_index) * 
                normalization_constant_fourth_ * 
                ComputeSingleIntegral_(mu_vec, nu_vec, rho_vec, sigma_vec);
            
            /*
            if (mu_index == 0 && nu_index == 1) {
              printf("this_integral = %g\n", this_integral);
            }
            */
            
            // if nodes are the same, both will be counted
            // Have to check nodes, not indices
            /*if (rho == sigma) {
              
              if (mu_index == 0 && nu_index == 1) {
                printf("considering %d, %d\n", rho_index, sigma_index);
              }*/
              integral_value = integral_value + this_integral;
            /*}*/
            // if nodes are different, then have to account for (sigma, rho)
            // pair as well
            /*else {
              
              if (mu_index == 0 && nu_index == 1) {
                printf("considering(double) %d, %d\n", rho_index, sigma_index);
              }
              integral_value = integral_value + 2 * this_integral;
            }*/
        
            // I should be able to compute rho/sigma and sigma/rho here as well    
            // It should only take figuring out some kind of four-way bound on 
            // the indices in the recursive call
           /* double old_rho_sigma = total_integrals_.ref(rho_index, sigma_index);
            old_rho_sigma = old_rho_sigma + 
                density_matrix_.ref(mu_index, nu_index) * this_integral;
                
            total_integrals_.set(rho_index, sigma_index, old_rho_sigma);
            total_integrals_.set(sigma_index, rho_index, old_rho_sigma);
            
            */
          } // sigma
          
        } // rho
        
        // Set both to account for mu/nu symmetry
        total_integrals_.set(mu_index, nu_index, integral_value);
        // Necessary to fill in the lower triangle
        if (mu != nu) {
          total_integrals_.set(nu_index, mu_index, integral_value);
        }
        
      } // nu
      
    } // mu
    
  } // ComputeIntegralsBaseCase_
  
  /**
   * Once an integral is approximated, this function fills in the approximate
   * value in the Fock matrix.
   * 
   * TODO: think about a smarter way to do this than loops (they won't save me 
   * much time.  Submatrices?
   */
  void FillApproximation_(IntegralTree* mu, IntegralTree* nu, 
                          IntegralTree* rho, IntegralTree* sigma,
                          double integral_approximation) {
  
  
  
    for (index_t mu_index = mu->begin(); mu_index < mu->end(); mu_index++) {
      
      for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
        
        double new_value = total_integrals_.ref(mu_index, nu_index) + 
            (rho->count() * sigma->count() * integral_approximation);
        
        // Set both to account for mu/nu symmetry
        total_integrals_.set(mu_index, nu_index, new_value);
        if (mu != nu) {
          total_integrals_.set(nu_index, mu_index, new_value);
        }
        
      } // nu
      
    } // mu
  
  } // FillApproximation_()
  
  /**
   * Handles the recursive part of the algorithm
   *
   * TODO: Figure out the recursive expansion pattern for four nodes
   *
   * TODO: make sure I take advantage of symmetry using some kind of a 
   * tree traversal
   */
  void ComputeIntegralsRecursion_(IntegralTree* mu, IntegralTree* nu, 
                                  IntegralTree* rho, IntegralTree* sigma) {
    
    
    // Need to figure out the right way to take advantage of the four-way 
    // symmetry
    /*if ((mu->stat().node_index() <= nu->stat().node_index()) &&
        (rho->stat().node_index() <= sigma->stat().node_index()) &&
        (((mu->stat().node_index() + nu->stat().node_index()) <= 
          (rho->stat().node_index() + sigma->stat().node_index())))) {
    */
   /* if ((mu->stat().node_index() <= nu->stat().node_index()) &&
        (rho->stat().node_index() <= sigma->stat().node_index())) {
     */ 
    if (mu->stat().node_index() <= nu->stat().node_index()) {
      
      //printf("Considering nodes: %d, %d, %d, %d\n", mu->stat().node_index(), 
            // nu->stat().node_index(), rho->stat().node_index(), sigma->stat().node_index());
      
      
      double integral_approximation;
      
      //Not sure if I should check for approximations or leaves first
      if (mu->is_leaf() && nu->is_leaf() && rho->is_leaf()
               && sigma->is_leaf()) {
        
        
        number_of_base_cases_++;
        ComputeIntegralsBaseCase_(mu, nu, rho, sigma);
        
      }      
      else if (CanApproximate_(mu, nu, rho, sigma, &integral_approximation)) {
        
        DEBUG_ASSERT(integral_approximation != BIG_BAD_NUMBER);
        
        number_of_approximations_++;
        
        FillApproximation_(mu, nu, rho, sigma, integral_approximation);
        
      }
      else {
        // find the node with greatest height
        // should also consider node with most points
        index_t mu_height = mu->stat().height();
        index_t nu_height = nu->stat().height();
        index_t rho_height = rho->stat().height();
        index_t sigma_height = sigma->stat().height();
        
        index_t greatest_height = mu_height;
        
        if (nu_height > greatest_height) {
          greatest_height = nu_height;
        }
        if (rho_height > greatest_height) {
          greatest_height = rho_height;
        }
        if (sigma_height > greatest_height) {
          greatest_height = sigma_height;
        }
        
        DEBUG_ASSERT(greatest_height > 0);
        
        if (greatest_height == mu_height) {
          
          DEBUG_ASSERT(!(mu->is_leaf()));
          
          ComputeIntegralsRecursion_(mu->left(), nu, rho, sigma);
          ComputeIntegralsRecursion_(mu->right(), nu, rho, sigma);
          
        }
        else if (greatest_height == nu_height) {

          DEBUG_ASSERT(!(nu->is_leaf()));
        
          ComputeIntegralsRecursion_(mu, nu->left(), rho, sigma);
          ComputeIntegralsRecursion_(mu, nu->right(), rho, sigma);
          
        }
        // Should consider prioritizing here, but maybe later
        else if (greatest_height == rho_height) {
        
          DEBUG_ASSERT(!(rho->is_leaf()));

          ComputeIntegralsRecursion_(mu, nu, rho->left(), sigma);
          ComputeIntegralsRecursion_(mu, nu, rho->right(), sigma);
        
        }
        else {
          
          DEBUG_ASSERT(greatest_height == sigma_height);
          
          DEBUG_ASSERT(!(sigma->is_leaf()));
          
          ComputeIntegralsRecursion_(mu, nu, rho, sigma->left());
          ComputeIntegralsRecursion_(mu, nu, rho, sigma->right());
          
        }
                
      }
          
    }
    
  } // ComputeIntegralsRecursion_
  
    
public:
  
  /**
   * Initialize the class with the centers of the data points, the fx module,
   * bandwidth
   */
  void Init(const Matrix& centers_in, struct datanode* mod, 
            const Matrix& density_in, const Matrix& core_in) {
  
    centers_.Copy(centers_in);
    
    module_ = mod;
    
    bandwidth_ = fx_param_double(module_, "bandwidth", 0.1);
    
    epsilon_ = fx_param_double(module_, "epsilon", 0.00001);
    
    number_of_approximations_ = 0;
    number_of_base_cases_ = 0;
    
    number_of_basis_functions_ = centers_.n_cols();
    fx_format_result(module_, "N", "%d", number_of_basis_functions_);
    
    // Not sure this is right, might want to consider starting with the previous
    // iteration's version
    total_integrals_.Copy(core_in);
    
    density_matrix_.Copy(density_in);
        
    leaf_size_ = fx_param_int(module_, "leaf_size", 30);
    
    tree_ = tree::MakeKdTreeMidpoint<IntegralTree>(centers_, leaf_size_, 
                &old_from_new_centers_, NULL);
    
    // Set up the indices of the nodes for symmetry pruning
    traversal_index_ = 0;
    PreOrderTraversal_(tree_);
    
    // The common normalization constant of all the Gaussians
    normalization_constant_fourth_ = pow((2 * bandwidth_ / math::PI), 3);
    
  } // Init
  
  /**
   * Resets the tree and parameters for the next Fock matrix computation
   */
  void ResetTrees(const Matrix& new_density) {

  } // ResetTrees
  
  /**
   * Drives the computation, assuming that all the parameters are correct
   */
  void ComputeFockMatrix() {
    
    ComputeIntegralsRecursion_(tree_, tree_, tree_, tree_);  
   // ComputeIntegralsBaseCase_(tree_, tree_, tree_, tree_);
    
  } // ComputeTwoElectronIntegrals
  
  /**
   * Returns the computed Fock matrix.  For now, it just prints it, but should 
   * eventually return it in a useable form for the SCF procedure.
   */
  void OutputFockMatrix() {
  
    //printf("number_of_approximations_ = %d\n", number_of_approximations_);
    //printf("number_of_base_cases_ = %d\n\n", number_of_base_cases_);
    fx_format_result(module_, "number_of_approximations", "%d", 
                     number_of_approximations_);
    fx_format_result(module_, "number_of_base_cases", "%d", 
                     number_of_base_cases_);
    //total_integrals_.PrintDebug();
  
  } // OutputFockMatrix()
  
 
}; // class DualTreeIntegrals



#endif 