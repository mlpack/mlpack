
#ifndef REGRESSION_PARENT_H
#define REGRESSION_PARENT_H

template <typename TKernel> class FastRegression{
public:
 
  //forward declaration of FastRegressionStat class
  class FastRegressionStat;

  //our tree uses FastRegressionStat
  typedef BinarySpaceTree < DHrectBound < 2 >, Matrix, FastRegressionStat > Tree;
 
  class FastRegressionStat {

  public:
    
    //The B^T B  matrix. this is required to be stored in the reference tree
    Matrix b_tb;

  
    /** lower bound on the B^TWB matrix for the query points.
     *  This is useful for only a query tree
     */
    
    Matrix b_twb_mass_l;

    /**
     * additional offset for the lower bound on the B^TWB  for the query
     * points owned by this node (for leaf nodes only).*/

    Matrix b_twb_more_l;
    
    /** lower bound offset Matrix passed from above
     *  This is required only by the query tree.
     *  Useful only for the query tree
     */
     
    Matrix b_twb_owed_l;

    
    /** upper bound on the B^TWB matrix for the query points owned by this node.
     * useful only for the query tree
     */
    
    Matrix b_twb_mass_u;
    
    /**
     * additional offset for the upper bound on the densities for the query
     * points owned by this node (for leaf nodes only).
     * Useful only for the query tree
     */
    Matrix b_twb_more_u;

    /** upper bound offset passed from above.
     *  This is useful only for the query tree
     */
    
    Matrix  b_twb_owed_u;


 
    //The B^T Y matrix that is required to be stored in the reference tree
    Matrix b_ty;
    

    /** lower bound on the B^TWY matrix for the query points.
     *  This is useful for only a query tree
     */
    
    Matrix b_twy_mass_l;

    /**
     * additional offset for the lower bound on the B^TWY  for the query
     * points owned by this node (for leaf nodes only).*/

    Matrix b_twy_more_l;
    
    /** lower bound offset Matrix passed from above
     *  This is required only by the query tree.
     *  Useful only for the query tree
     */
     
    Matrix b_twy_owed_l;

    
    /** upper bound on the B^TWY matrix for the query points owned by this node.
     * useful only for the query tree
     */
    
    Matrix b_twy_mass_u;
    
    /**
     * additional offset for the upper bound on the densities for the query
     * points owned by this node (for leaf nodes only).
     * Useful only for the query tree
     */
    Matrix b_twy_more_u;

    /** upper bound offset passed from above.
     *  This is useful only for the query tree
     */
    
    Matrix  b_twy_owed_u;
    

    //These are the Init functions of FastRegressionStat
    void Init ();

    /**This is the Init function of the leaf*/

    void Init (const Matrix & dataset, index_t & start, index_t & count);
    /** This is the Init function of an internal node */
    
    void Init (const Matrix & dataset, index_t & start, index_t & count,
	       const FastRegressionStat &left_stat, 
	       const FastRegressionStat &right_stat);

    FastRegressionStat ();

    ~FastRegressionStat ();
  };


 //public members of the class FastMatrix



  //So lets create an enum that will tell us whether we should prune
  //both the matrix B^TWB and the vector B^TWY or none of them or
  //just one of them
  typedef enum prune_t{

    PRUNE_BOTH,
    PRUNE_B_TWB,
    PRUNE_B_TWY,
    PRUNE_NOT,
  }prune_t;


  //We need another enum to tell us to check for prunability of
  //B^TWB or B^TWY or both

  typedef enum check_for_prune_t{
 
    CHECK_FOR_PRUNE_BOTH,
    CHECK_FOR_PRUNE_B_TWB,
    CHECK_FOR_PRUNE_B_TWY,
  }check_for_prune_t;


  typedef enum CRITERIA_FOR_PRUNE_t{
 
    CRITERIA_FOR_PRUNE_COMPONENT,
    CRITERIA_FOR_PRUNE_FNORM,
  }criteria_for_prune_t;

 //Private member of FastMatrix
 private:

  //Lets declare private variables
 
  /**The criteria for pruning */
  criteria_for_prune_t pruning_criteria;
  /** query dataset */
  Matrix qset_;
 
  /**reference dataset */
  Matrix rset_;
 
  /** query tree */
  Tree *qroot_;
 
  /** reference tree */
  Tree *rroot_;
 
  /** list of kernels to evaluate */
  TKernel kernel_;
 
  /* lower bound on the B^TWB matrix*/
 
  ArrayList <Matrix> b_twb_l_estimate_;
 
  /* Estimate for the B^TWB matrix for each query point*/
  ArrayList <Matrix> b_twb_e_estimate_;
 
  /* upper bound on the B^TWB matrix*/
  ArrayList <Matrix> b_twb_u_estimate_;


  /* lower bound on the B^TWY matrix*/
 
  ArrayList <Matrix> b_twy_l_estimate_;
 
  /* Estimate for the B^TWY matrix for each query point*/
  ArrayList <Matrix> b_twy_e_estimate_;
 
  /* upper bound on the B^TWY matrix*/
  ArrayList <Matrix> b_twy_u_estimate_;
 
  /** accuracy parameter */
  double tau_;

  /** Set of weights for the reference points */
  Vector rset_weights_;

  /** Regression estimates of the query points */
  Vector regression_estimate_;

  /** Mappings from old dataset to new dataset
   * Remember that when we build the query tree and the
   * reference tree out of the query and reference
   * datasets, the datasets get permuted. These
   * arrays give a bidirectional mapping.
   */

  ArrayList <int> old_from_new_r_;
  ArrayList <int> new_from_old_r_;
  //Intersting *Private* functions....................

  void ScaleDataByMinmax_();

 /** This function finds the order in which a node nd should
   * be pruned when being comparee with nd1 and nd2. It is simply
   * based on the distance metric. Hence if nd1 is neared to node nd
   * then we investigate nd and nd1 else we investigate nodes nd and nd2

  */
  void BestNodePartners_ (Tree * nd, Tree * nd1, Tree * nd2,
			  Tree ** partner1,Tree ** partner2);


 // So lets preprocess the tree. We shall allocate memory for  all the variables
  // defined in the statistic of the tree and initialize them to proper values
  void PreProcess_(Tree *node, Matrix &dataset);


 void FlushOwedValues_(Tree *qnode, check_for_prune_t flag);
   
 void UpdateBounds_(Tree *qnode,  Matrix &dl_b_twb, 
		     Matrix &du_b_twb, Matrix &dl_b_twy, Matrix &du_b_twy, 
		    check_for_prune_t flag);

 prune_t Prunable_(Tree *, Tree *, Matrix &, Matrix &du,
		   Matrix &dl, Matrix &, check_for_prune_t);

 void FRegressionBase_(Tree *qnode, Tree *rnode, check_for_prune_t flag);

 void MergeChildBounds_(Tree *,check_for_prune_t flag);


 void CallRecursively_(Tree *qnode, Tree *rnode,check_for_prune_t flag);

 void FRegression_(Tree *qnode, Tree *rnode, check_for_prune_t flag);

 void  UpdateBoundsForPruningB_TWB_(Tree *qnode, Matrix &dl_b_twb, 
				    Matrix &du_b_twb);
 
 void  UpdateBoundsForPruningB_TWY_(Tree *qnode, Matrix &dl_b_twy, 
				   Matrix &du_b_twy);

 void  CalculateB_TYRecursively_(Tree *rnode);
 
 void SetUpperBounds_(Tree *node);
 
 index_t PrunableB_TWY_(Tree *,Tree *,Matrix &dl,Matrix &du);

 index_t  PrunableB_TWB_(Tree *, Tree *, Matrix &, Matrix &);
 
 void MergeChildBoundsB_TWB_(FastRegressionStat *left_stat, 
			     FastRegressionStat *right_stat, 
			     FastRegressionStat &parent_stat);

 void MergeChildBoundsB_TWY_(FastRegressionStat *left_stat, 
			     FastRegressionStat *right_stat, 
			     FastRegressionStat &parent_stat);

 void FRegressionBaseB_TWB_(Tree *, Tree *);
 void FRegressionBaseB_TWY_(Tree *, Tree *);
 void PostProcess_(Tree *qnode);
 void Print_();
 void ObtainRegressionEstimate_();
 void PrintRegressionEstimate_();
 double SquaredFrobeniusNorm_(Matrix &);
 double Compute1NormLike_(Matrix &);
 
 public:
 
  /* getter functions */

 Vector & get_regression_estimate(); 
 Matrix & get_query_dataset();
 Matrix& get_reference_dataset();
 Matrix &  get_b_twy_estimates(index_t );
 Matrix &  get_b_twb_estimates(index_t );

  /** This function will be called after update bounds has been called in the 
   *  function Prunable. This will flush the owed values as they have already 
   * been incorporated in the mass values by the function Update Bounds

  */

  ArrayList<index_t>& FastRegression<TKernel>::get_old_from_new_r();
 
  void Compute();

  void Init(Matrix &q_matrix, Matrix &r_matrix, double bandwidth,
	  double tau,index_t leaf_length,Vector &rset_weights, char *);
};



/** This class contains functions to calculate B^TWY and B^TWB
    naively
  */
template < typename TKernel > class NaiveCalculation{

 private:

  /** query dataset */
  Matrix qset_;

  /** reference dataset */
  Matrix rset_;

  /** kernel */
  TKernel kernel_;

  /** computed Matrices */
  ArrayList <Matrix > b_twy_naive_estimate_;

  /**computed Matrix */
 /** computed Matrices */
  ArrayList <Matrix > b_twb_naive_estimate_;

  /**Reference weights. these are nothing but the regression values
    of the reference points. These are used in forming the Y vector
  */

  Vector rset_weights_;

  Vector regression_estimate_;

  /** Remember that we first do the the fast vector calculations and 
   *  then pass on the datasets to the naive method for naive calculations.
   *  when we do so we are actually permuting the dataset.Hence we need to know the 
   *  permutation of the dataset so that we can use the dataset in the right way.
   *  This permutation is captured by old_from_new_r_. This is a mapping from the new to the 
   *  old reference set
   */

  ArrayList<index_t> old_from_new_r_;


 public:

  //getters...............

  double get_vector_estimates(index_t, index_t );
  
 /** This is the function which builds up the B^TWY 
  * vector by usimng matrix calculations. Note this can be 
  *  optimized by using fastlib's Lapack functions 
  */
   
  void PrintRegressionEstimate_(Vector &);
  void ObtainRegressionEstimate_();
  void Compute ();
  
  void Init (Matrix &, Matrix &, ArrayList<index_t> &,double ,Vector&);
  void Print_();

  void ComputeMaximumRelativeError(ArrayList<Matrix> &, ArrayList<Matrix> &, char *);
  double SquaredFrobeniusNorm_(Matrix &);

  void CompareFastWithNaive(Vector &);
};

#include "regression_ll1.h"
#include "regression_ll2.h"
#include "regression_ll_naive.h"

#endif
