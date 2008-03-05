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
  
  DualTreeIntegrals() {}
  
  ~DualTreeIntegrals() {}
  
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
    
   public:
    void Init() {
      
      node_index_ = -1;
      min_bandwidth_ = 0.0;
      max_bandwidth_ = DBL_MAX;
      
    } // Init
    
    void Init(const Matrix& matrix, index_t start, index_t count) {
      
      Init();
      
    } // Init (leaves)
    
    void Init(const Matrix& matrix, index_t start, index_t count, 
              const IntegralStat& left, const IntegralStat& right) {
    
      Init();
      
    } // Init (non-leaves)
    
    index_t node_index() {
      
      return node_index_;
      
    } // node_index
    
    void set_node_index(index_t new_index) {
      
      node_index_ =  new_index;
      
    } // set_node_index
    
    double min_bandwidth() {
      
      return min_bandwidth_; 
    
    } // min_bandwdith
    
    void set_min_bandwidth(double new_min) {
      
      min_bandwidth_ = new_min; 
    
    } //set_min_bandwidth
    
    void set_max_bandwidth(double new_max) {
      
      max_bandwidth_ = new_max;
      
    } // set_max_bandwidth
    
    double max_bandwidth() {
      
      return max_bandwidth_;
      
    } // max_bandwidth_
    
  }; // class IntegralStat
  
  // This assumes identical bandwidth small Gaussians
  // Otherwise, I'll need something other than a Matrix
  // I should also consider something better than bounding boxes
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, IntegralStat> IntegralTree; 
  
private:
    
  // The tree 
  IntegralTree* tree_;
  
  // The centers of the identical, spherical Gaussian basis functions
  Matrix centers_;
  
  // The fx module
  struct datanode* module_;
  
  // The common bandwidth of all the basis functions
  double bandwidth_;
  
  // The number of times an approximation is invoked
  int number_of_approximations_;
  
  // The value eps where the returned integrals are within 1-eps of the true 
  // value
  double epsilon_;
  
  // The return values are stored here
  // total_integrals_.ref(i, j) is the fock matrix entry i, j
  Matrix total_integrals_;
  
  // The density matrix, will be input for now
  Matrix density_matrix_;
  
  // The total number of basis functions
  index_t number_of_basis_functions_;
  
  index_t traversal_index_;
  

  /////////////////////// Functions ///////////////////////////////////
  
  void PreOrderTraversal_(IntegralTree* this_node) {
    
    if (is_leaf(this_node)) {
     
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
    
    /*
    * don't think I actually need these 
    double four_way_min_dist = 
        0.25 * (mu->bound().MinDistanceSq(rho->bound()) + 
                nu->bound().MinDistanceSq(sigma->bound()));
    
    double four_way_max_dist = 
        0.25 * (mu->bound().MaxDistanceSq(rho->bound()) + 
                nu->bound().MaxDistanceSq(sigma->bound()));
    
    // Not sure this one is correct
    double four_way_mid_dist = 
        0.25 * (mu->bound().MidDistanceSq(rho->bound()) + 
                nu->bound().MidDistanceSq(sigma->bound()));
    */
    
    double up_bound = ComputeSingleIntegral_(mu_nu_min_dist, rho_sigma_min_dist, 
        mu_rho_min_dist, nu_sigma_min_dist);
    double low_bound = ComputeSingleIntegral_(mu_nu_max_dist, 
        rho_sigma_max_dist, mu_rho_max_dist, nu_sigma_max_dist);
    
    DEBUG_ASSERT(up_bound >= low_bound);
    
    /* the total error I'm allowed to make here is epsilon *  */
    double my_allowed_error = epsilon_ * rho->count() * sigma->count()
        / (number_of_basis_functions_ * number_of_basis_functions_);
    
    double my_max_error = 0.5 * (up_bound - low_bound);
    
    if (my_max_error < my_allowed_error) {
      
      DEBUG_ASSERT(my_max_error < epsilon_);
      
      can_prune = true;
      
      // I don't think this is right
      epsilon_ = epsilon_ - my_max_error;
      
      DEBUG_ASSERT(epsilon_ > 0.0);
      
      *approximate_value = my_max_error;
      
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
   *
   * BUG: What about z == 0 ? 
   */
  double ErfLikeFunction_(double z) {
    
    DEBUG_ASSERT(z != 0);
    
    return((1/sqrt(z)) * sqrt(math::PI) * 0.5 * erf(sqrt(z)));
    
  } // ErfLikeFunction_
  
  /**
   * Computes a single integral based on the distances between the centers.  
   * Intended for use in bounding.  
   *
   * BUG: I don't think this works if mu == rho and nu == sigma.  
   */
  double ComputeSingleIntegral_(double mu_nu_dist, double rho_sigma_dist, 
                                     double mu_rho_dist, double nu_sigma_dist) {
    
    double return_value;
    
    return_value = 0.25 * pow(math::PI, 2.5);
    
    return_value = return_value * ErfLikeFunction_(bandwidth_ * 0.25 * 
                                                 (mu_rho_dist + nu_sigma_dist));
    
    return_value = return_value * 
        exp(-1.0 * bandwidth_ * (mu_nu_dist + rho_sigma_dist));
    
    return return_value;
    
  } // ComputeSingleIntegral_
  
  /**
   * Finds the single integral of four Gaussians at the given centers
   *
   * TODO: decide if I'm computing the ERI and the exchange integral both here, 
   * or each in a separate function
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
    double return_value = 0.25 * pow(math::PI, 2.5);
    //double return_value = math::Pow<math::PI, (4 * bandwidth_)>(2.5);
    
    Vector mu_nu_center;
    Vector rho_sigma_center;
    
    la::AddInit(mu_center, nu_center, &mu_nu_center);
    la::AddInit(rho_center, sigma_center, &rho_sigma_center);
    
    // Assuming equal bandwidths here
    la::Scale(0.5, &mu_nu_center);
    la::Scale(0.5, &rho_sigma_center);
    
    // F(\alpha d^2(x_{i j} x_{k l}))
    return_value = return_value * 
      ErfLikeFunction_(bandwidth_ * 
                      la::DistanceSqEuclidean(mu_nu_center, rho_sigma_center));
    
    
    // exp(-\alpha (d^2(x_i, x_j) - d^2(x_k, x_l)
    double between_centers_dists = 
        la::DistanceSqEuclidean(mu_center, nu_center) 
        + la::DistanceSqEuclidean(rho_center, sigma_center);
    
    //return_value = return_value * 
       // pow(math::E, 
         //   (-1 * bandwidth_ * between_centers_dists));
    return_value = return_value * exp(-1 * bandwidth_ * between_centers_dists);
        
    return return_value;
    
  } // ComputeSingleIntegral_
  
  /**
   * Exhaustively computes the total integrals among the four nodes 
   *
   * TODO: account for symmetry here, particularly the mu+nu/rho+sigma kind
   */
  void ComputeIntegralsBaseCase_(IntegralTree* mu, IntegralTree* nu, 
                                 IntegralTree* rho, IntegralTree* sigma) {
    
    for (index_t mu_index = mu->begin(); mu_index < mu->end(); mu_index++) {
     
      for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
       
        double integral_value = 0.0;
        
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
            
            // Multiply by two to account for rho/sigma symmetry 
            integral_value = integral_value + 
                2 * ComputeSingleIntegral_(mu_vec, nu_vec, rho_vec, sigma_vec);
            
          } // sigma
          
        } // rho
        
        // Set both to account for mu/nu symmetry
        total_integrals_.set(mu_index, nu_index, integral_value);
        // I think this is right
        // I should also just consider outputting the symmetric matrix
        total_integrals_.set(nu_index, mu_index, integral_value);
        // I need to give more thought to the symmetry with rho and sigma
        
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
                          IntegralTree* rho, IntegralTree* sigma
                          double integral_approximation) {
  

    
    }
  
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
    
    if ((mu->stat().node_index() <= nu->stat().node_index()) &&
        (rho->stat().node_index() <= sigma->stat().node_index()) &&
        (((mu->stat().node_index() + nu->stat().node_index()) <= 
          (rho->stat().node_index() + sigma->stat().node_index())))) {
    
      
      double integral_approximation;
      
      if (CanApproximate_(mu, nu, rho, sigma, &integral_approximation)) {
        
        DEBUG_ASSERT(integral_approximation != BIG_BAD_NUMBER);
        
        number_of_approximations_++;
        
        FillApproximation_(mu, nu, rho, sigma, integral_approximation);
        
      }
      else if (mu->is_leaf() && nu->is_leaf() && rho->is_leaf()
               && sigma->is_leaf()) {
        
        ComputeIntegralsBaseCase_(mu, nu, rho, sigma);
        
      }
      else {
        
        // Need to figure out four-way recursion here
        
        ComputeIntegralsRecursion_(mu->left(), nu->left(), rho->left(), sigma->left());
        ComputeIntegralsRecursion_(mu->left(), nu->left(), rho->left(), sigma->right());
        ComputeIntegralsRecursion_(mu->left(), nu->left(), rho->right(), sigma->left());
        ComputeIntegralsRecursion_(mu->left(), nu->left(), rho->right(), sigma->right());

        ComputeIntegralsRecursion_(mu->left(), nu->right(), rho->left(), sigma->left());
        ComputeIntegralsRecursion_(mu->left(), nu->right(), rho->left(), sigma->right());
        ComputeIntegralsRecursion_(mu->left(), nu->right(), rho->right(), sigma->left());
        ComputeIntegralsRecursion_(mu->left(), nu->right(), rho->right(), sigma->right());
        
        ComputeIntegralsRecursion_(mu->right(), nu->left(), rho->left(), sigma->left());
        ComputeIntegralsRecursion_(mu->right(), nu->left(), rho->left(), sigma->right());
        ComputeIntegralsRecursion_(mu->right(), nu->left(), rho->right(), sigma->left());
        ComputeIntegralsRecursion_(mu->right(), nu->left(), rho->right(), sigma->right());
        
        ComputeIntegralsRecursion_(mu->right(), nu->right(), rho->left(), sigma->left());
        ComputeIntegralsRecursion_(mu->right(), nu->right(), rho->left(), sigma->right());
        ComputeIntegralsRecursion_(mu->right(), nu->right(), rho->right(), sigma->left());
        ComputeIntegralsRecursion_(mu->right(), nu->right(), rho->right(), sigma->right());
        
      } 
      
    }
    
  } // ComputeIntegralsRecursion_
  
    
public:
  
  /**
   * Initialize the class with the centers of the data points, the fx module,
   * bandwidth
   */
  void Init(const Matrix& centers_in, struct datanode* mod, double band, 
            double error, const Matrix& density_in) {
  
    centers_.Copy(centers_in);
    
    module_ = mod;
    
    bandwidth_ = band;
    
    epsilon_ = error;
    
    number_of_approximations_ = 0;
    
    number_of_basis_functions_ = centers_.n_cols();
    
    // Not sure this is right, might want to consider starting with the previous
    // iteration's version
    total_integrals_.Init(3, number_of_basis_functions_);
    
    density_matrix_.Copy(density_in);
    
    traversal_index_ = 0;
    
  } // Init
  
  
  void ComputeTwoElectronIntegrals() {
    
    
    
  } // ComputeTwoElectronIntegrals
  
 
}; // class DualTreeIntegrals



#endif 