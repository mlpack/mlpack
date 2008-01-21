#ifndef REGRESSION_VECTOR_H
#define REGRESSION_VECTOR_H
#include "values.h"
#include "fastlib/fastlib_int.h"

/** In this code we shall evaluate B^TWY for each query point. 
 *  this is a vector of length D+1, where D is the dimensionality 
 *  of the dataset. We shall approach this problem in 2 ways.
 *  We sall try to evaluate this vector by a naive method and by
 *  using a dual tree approach 
 */

/** This is the naive approach where we try to estimate
 *  B^TWY by naive method of matrix calculation
 */

/** We define the NaiveVectorCalculation which supports functions
 *  to calculate the vector B^TWY in a naive way
 */

template < typename TKernel > class NaiveVectorCalculation{

 private:

  /** query dataset */
  Matrix qset_;

  /** reference dataset */
  Matrix rset_;

  /** kernel */
  TKernel kernel_;

  /** computed vectors */
  ArrayList < Vector > vector_results_;

  /**Reference weights. these are nothing but the regression values
    of the reference points. These are used in forming the Y vector
  */

  Vector rset_weights_;

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

  double get_vector_estimates(index_t i, index_t j){

    return vector_results_[i][j];
  }

 
  /** This is the function which builds up the B^TWY 
   * vector by usimng matrix calculations. Note this can be 
   *  optimized by using fastlib's Lapack functions 
   */
   
  void Compute (){

    // compute unnormalized sum
    for (index_t q = 0; q < qset_.n_cols (); q++){	//for each query point
	
      const double *q_col = qset_.GetColumnPtr (q);
      for (index_t d = 0; d < rset_.n_rows () + 1; d++){	//along each direction
	for (index_t r = 0; r < rset_.n_cols (); r++){	//for each reference point
	  if (d == 0){
	   
	    const double *r_col = rset_.GetColumnPtr (r);
	    double dsqd =la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	    double val = kernel_.EvalUnnormOnSq (dsqd);
	    val *= rset_weights_[old_from_new_r_[r]];
	    vector_results_[q][d] += val;
	  }
	  else{

	    const double *r_col = rset_.GetColumnPtr (r);
	    double dsqd =
	      la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	    double val = kernel_.EvalUnnormOnSq (dsqd);
	    val *= rset_weights_[old_from_new_r_[r]];
	    val *= rset_.get (d - 1, r);
	    vector_results_[q][d] += val;
	  }

	}
      }
    }
   
  }

  void Init (){

    vector_results_.Init (qset_.n_cols ());
    for (index_t i = 0; i < qset_.n_cols (); i++){
      vector_results_[i].Init (rset_.n_rows () + 1);
      vector_results_[i].SetAll (0);
    }
  }

  void Init (Matrix & qset, Matrix & rset, ArrayList<index_t> &old_from_new_r){
    
    // get datasets
    qset_.Alias (qset);
    rset_.Alias (rset);
    
    //get permutation
    old_from_new_r_.Copy(old_from_new_r);

    // get bandwidth
    kernel_.Init (fx_param_double_req (NULL, "bandwidth"));
   

    vector_results_.Init (qset_.n_cols ());
  
    for (index_t i = 0; i < qset_.n_cols (); i++){
      vector_results_[i].Init (rset_.n_rows () + 1);
      vector_results_[i].SetZero ();
    }

      

    // read the reference weights.
    char *rwfname = NULL;

    if (fx_param_exists (NULL, "dwgts")){
      //rwfname is the filename having the refernece weights
      rwfname = (char *) fx_param_str (NULL, "dwgts", NULL);
    }

    if (rwfname != NULL){
      Dataset ref_weights;
      ref_weights.InitFromFile (rwfname);
      rset_weights_.Copy (ref_weights.matrix ().GetColumnPtr (0), ref_weights.matrix ().n_rows ());	//Note rset_weights_ is a vector of weights
    }

    else{
      rset_weights_.Init (rset_.n_cols ());
      rset_weights_.SetAll (1);

    }
     
  }

  /*  void PrintDebug (){

    FILE *fp;
    const char *fname;

    if (fx_param_exists (NULL, "naive_kde_output")){

      fname = fx_param_str (NULL, "naive_kde_output", NULL);
      fp = fopen (fname, "w+");
    }
    for (index_t q = 0; q < qset_.n_cols (); q++){
      
      fprintf(fp,"Point:");
      for(index_t z=0;z<qset_.n_rows();z++){
	fprintf(fp,"%f",qset_.get(z,q));
	fprintf(fp," ");
      }
      fprintf(fp,"\n");
      for (index_t d = 0; d < rset_.n_rows () + 1; d++){
	fprintf (fp, "%f\n", densities_[q][d]);
      }
    }
    }*/

  /* void ComputeMaximumRelativeError (ArrayList < Vector > density_estimate) {

    double max_rel_err = 0;
    for (index_t q = 0; q < qset_.n_cols (); q++){
      for (index_t d = 0; d < rset_.n_rows () + 1; d++){

	double rel_err = fabs (density_estimate[q][d] - densities_[q][d]) / densities_[q][d];

	if (rel_err > max_rel_err){
	  max_rel_err = rel_err;
	}
      }
    }

    fx_format_result (NULL, "maxium_relative_error_for_fast_KDE", "%g",
    max_rel_err);
    }*/

};  //Class NaiveKDE ends........



template < typename TKernel > class FastVectorCalculation{
 public:

  // forward declaration of FastVectorStat class
  class FastVectorStat;

  // our tree type using the FastVectorStat
  typedef BinarySpaceTree < DHrectBound < 2 >, Matrix, FastVectorStat > Tree;

  class FastVectorStat{

  public:

    /** This is the weights dimension wise. $\sum (x_{ij}y_{i})$. 
     *  It is a D+1 dimensional array
     */

    Vector weight_of_dimension;

    /** lower bound on the Vector estimate for the query points owned by this node . 
     *  Realise that  it is a 1-D vector for each node in the tree
     */
    Vector mass_l;

    /**
     * additional offset for the lower bound on the vector estimates for the query
     * points owned by this node (for leaf nodes only). It is a 1-D vector
     */
    Vector more_l;

    /**
     * lower bound offset passed from above. Realise that now it is a 1-D vector
     */
    Vector owed_l;

    /** upper bound on the vector estimates for the query points owned by this node. 
     *  Realise that now it is a 1-D vector
     */
    Vector mass_u;

    /**
     * additional offset for the upper bound on the vector estimates  for the query
     * points owned by this node (for leaf nodes only). 
     * Realise that now it is a 1-D vector
     */
    Vector more_u;

    /**
     * upper bound offset passed from above. 
     */
    Vector owed_u;


    //These are the Init functions of FastVectorStat
    void Init (){

    }

    void Init (const Matrix & dataset, index_t & start, index_t & count){

      Init ();
    }

    void Init (const Matrix & dataset, index_t & start, index_t & count,
	       const FastVectorStat & left_stat, const FastVectorStat & right_stat){
      Init ();
    }

    /** So this function is called whenever the mass_l and mass_u estimates of 2 nodes
     * have been changed
     */

    void MergeChildBounds (FastVectorStat & left_stat, FastVectorStat & right_stat,
			   index_t dim){
      // improve lower and upper bound
      for (index_t i = 0; i < dim; i++)
	{
	  //mass_l for parent= max(parent, min(children))
	  mass_l[i] =
	    max (mass_l[i], min (left_stat.mass_l[i], right_stat.mass_l[i]));

	  //mass_u for parent=min(parent,max(children))

	  mass_u[i] =
	    min (mass_u[i], max (left_stat.mass_u[i], right_stat.mass_u[i]));
	}
    }

    FastVectorStat (){
    }

    ~FastVectorStat (){
    }

  };				//WITH THIS CLASS FastVectorStat COMES TO AN END

  //Private member of FastVectorCalculation
 private:

  /**Number of prunes*/
  int number_of_prunes;

  /** query dataset */
  Matrix qset_;

  /**reference dataset */
  Matrix rset_;

  /** query tree */
  Tree *qroot_;


  /** reference tree */
  Tree *rroot_;

  /** reference weights */
  Vector rset_weights_;

  /** list of kernels to evaluate */
  TKernel kernel_;

  /** lower bound on the vector_estimate.This will now be a 2-D vector*/
  ArrayList <Vector> vector_estimate_l_;

  /** vector_estimate computed .This will now be a 2-D vector*/
  ArrayList <Vector> vector_estimate_e_;

  /** upper bound on the vector_estimate.This will now be a 2-D vector */
  ArrayList <Vector> vector_estimate_u_;

  /** accuracy parameter */
  double tau_;

  /**Number of base case operations*/
  int fast_kde_base;

  /** As mentioned earlies the following 2 arrays keep track of the permutation
   * of the reference dataset
   */
  ArrayList <int> new_from_old_r_;
  ArrayList<int> old_from_new_r_;
 
  /** For numerical stability of our calculations we scaled the entire dataset
   *  to 0-1 range by using min/max values
   */

  void scale_data_by_minmax (){

    int num_dims = rset_.n_rows ();
    DHrectBound < 2 > qset_bound;
    DHrectBound < 2 > rset_bound;
    qset_bound.Init (qset_.n_rows ());
    rset_bound.Init (qset_.n_rows ());

    // go through each query/reference point to find out the bounds
    for (index_t r = 0; r < rset_.n_cols (); r++){
      Vector ref_vector;
      rset_.MakeColumnVector (r, &ref_vector);
      rset_bound |= ref_vector;
    }
    for (index_t q = 0; q < qset_.n_cols (); q++){
      Vector query_vector;
      qset_.MakeColumnVector (q, &query_vector);
      qset_bound |= query_vector;
    }

    for (index_t i = 0; i < num_dims; i++){
      DRange qset_range = qset_bound.get (i);
      DRange rset_range = rset_bound.get (i);
      double min_coord = min (qset_range.lo, rset_range.lo);
      double max_coord = max (qset_range.hi, rset_range.hi);
      double width = max_coord - min_coord;
      
      for (index_t j = 0; j < rset_.n_cols (); j++){
	rset_.set (i, j, (rset_.get (i, j) - min_coord) / width);
      }

      if (strcmp (fx_param_str (NULL, "query", NULL),
		  fx_param_str_req (NULL, "data"))){
	for (index_t j = 0; j < qset_.n_cols (); j++){
	  qset_.set (i, j, (qset_.get (i, j) - min_coord) / width);
	}
      }
    }
  }

  // member functions
  
  /** Preprocess the tree. this is where we  allocate memory for all the variables defined
   *  in the statistic of the tree, and set their initial values appropriately 
   */ 
  
  void PreProcess (Tree * node){
    
    /** The approach is to estimate the D+1 length vector 
     *  component by component.We do that by estimating each
     *  each element by weighted kernel density kind of estimation
     */

    node->stat ().mass_l.Init (rset_.n_rows() + 1);  //Mass_l initialized
    node->stat ().owed_l.Init (rset_.n_rows() + 1);  //owed_l initialized
    node->stat ().owed_u.Init (rset_.n_rows() + 1);  //owed_u initialized
    node->stat ().weight_of_dimension.Init (rset_.n_rows() + 1); //weight_of_dimension initalized
    
    
    // initialize lower bound to 0

    for (int i = 0; i < rset_.n_rows () + 1; i++){
      node->stat ().mass_l[i] = 0;
      node->stat ().owed_l[i] = 0;
      node->stat ().owed_u[i] = 0;
    }

    //Base CAse.....
    
    if (node->is_leaf ()){
      
      //This is a leaf node.......................
      node->stat ().more_l.Init (rset_.n_rows () + 1);
      node->stat ().more_u.Init (rset_.n_rows () + 1);
      node->stat ().more_l.SetAll (0);
      node->stat ().more_u.SetAll (0);
      node->stat ().weight_of_dimension.SetZero ();
      
      for (int i = 0; i < rset_.n_rows () + 1; i++){ //along all dimensions
	for (int j = node->begin (); j < node->end (); j++){ //looping over all points
	  if (i != 0){
	    node->stat ().weight_of_dimension[i] +=
	      rset_.get (i - 1, j) * rset_weights_[old_from_new_r_[j]];
	  }
	  else{
	    node->stat ().weight_of_dimension[i] += rset_weights_[old_from_new_r_[j]];
	    
	  }
	}
      }
    }
      
    
    // for non-leaf node, recurse
    else{
    
      
      PreProcess (node->left ());
      PreProcess (node->right ());

      for (int count = 0; count < rset_.n_rows () + 1; count++){
	
	node->stat ().weight_of_dimension[count] =
	  node->left ()->stat ().weight_of_dimension[count] +
	  node->right ()->stat ().weight_of_dimension[count];
      }
      
    }
    
  }

  /** This function updates the mass_l and mass_u bounds of the parent node
   * 
   */

  void UpdateBounds (Tree * qnode, Vector dl, Vector du){

    // query self statistics
    FastVectorStat & qstat = qnode->stat ();
    
    // Changes the upper and lower bounds.
    
    for (int i = 0; i < rset_.n_rows () + 1; i++){ //along each dimension
      //printf("Will add dl=%f and du=%f\n",dl[i],du[i]);
      qstat.mass_l[i] += dl[i];
      qstat.mass_u[i] += du[i];
    }

    // for a leaf node, incorporate the lower and upper bound changes into
    // its additional offset

    if (qnode->is_leaf ()){
      for (int t = 0; t < rset_.n_rows () + 1; t++){
	qstat.more_l[t] += dl[t];
	qstat.more_u[t] += du[t];
      }
    }

    // otherwise, incorporate the bound changes into the owed slots of
    // the immediate descendants
    else{

      for (int i = 0; i < rset_.n_rows () + 1; i++){
	//transmission of the owed values to the children
	qnode->left ()->stat ().owed_l[i] += dl[i];
	qnode->left ()->stat ().owed_u[i] += du[i];
	  
	qnode->right ()->stat ().owed_l[i] += dl[i];
	qnode->right ()->stat ().owed_u[i] += du[i];
      }
    }
  
  }

  /** So we come here when we get to 2 leaves which cannot be 
   *  pruned. hence for these pair of nodes we calculate the
   *  vector by exhaustive calulations over the rnode. this is 
   *  not a very costly calculation. However the code is still not 
   *  optimized, and can be optimized by using fastlib's Lapack utilities
   */

  void FVectorEstimateBase (Tree * qnode, Tree * rnode){

    //subtract along each component because now you are doing exhaustive computation

    for(index_t d=0;d<rset_.n_rows()+1;d++){ //along each dimension

      qnode->stat().more_u[d]-=rnode->stat().weight_of_dimension[d];
    }
     
    for (index_t q = qnode->begin (); q < qnode->end (); q++){ //for each query node

      // get query point
      const double *q_col = qset_.GetColumnPtr (q);

      for (index_t i = 0; i < rset_.n_rows () + 1; i++){	//Along all dimensions

	for (index_t r = rnode->begin (); r < rnode->end (); r++){ // get reference point

	  const double *r_col = rset_.GetColumnPtr (r);

	  // pairwise distance and kernel value
	  double dsqd =
	    la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	  double ker_value = kernel_.EvalUnnormOnSq (dsqd);
	  if (i != 0){
	    vector_estimate_l_[q][i] +=
	      ker_value * rset_weights_[old_from_new_r_[r]] * rset_.get (i - 1, r);
	    vector_estimate_u_[q][i] +=
	      ker_value * rset_weights_[old_from_new_r_[r]] * rset_.get (i - 1, r);
	  }
	  else{
	    vector_estimate_l_[q][i] += ker_value * rset_weights_[old_from_new_r_[r]];
	    vector_estimate_u_[q][i] += ker_value * rset_weights_[old_from_new_r_[r]];

	  }
	}
      }
    }

    // get a tighter lower and upper bound for every dimension by looping over each query point
   
    Vector min_l;
    min_l.Init (rset_.n_rows () + 1);

    Vector max_u;
    max_u.Init (rset_.n_rows () + 1);
    min_l.SetAll (DBL_MAX);
    max_u.SetAll (DBL_MIN);


    //Construct a min vector by going along each component and finding out the least value
    //Construct a max vector by going along each component and finding out the maximum value

    for (index_t t = 0; t < rset_.n_rows () + 1; t++){
      for (index_t q = qnode->begin (); q < qnode->end (); q++){
	if (vector_estimate_l_[q][t] < min_l[t]){

	  min_l[t] = vector_estimate_l_[q][t];
	}
	if (vector_estimate_u_[q][t] > max_u[t]){
	  max_u[t] = vector_estimate_u_[q][t];
	}
      }
    }


    // tighten lower and upper bound
    for (index_t i = 0; i < rset_.n_rows () + 1; i++){
      qnode->stat ().mass_l[i] = min_l[i] + qnode->stat ().more_l[i];
      qnode->stat ().mass_u[i] = max_u[i] + qnode->stat ().more_u[i];
    }
  }


  /** 
   * checking for prunability of the query and the reference pair
   * 
   */

  /** We check for punability along each direction and prune only if it is prunable 
   *  along each direction
   */

  int Prunable (Tree * qnode, Tree * rnode, DRange & dsqd_range,
		DRange & kernel_value_range, Vector &dl, Vector &du){

    // query node stat
    FastVectorStat & stat = qnode->stat ();

    // try pruning after bound refinement: first compute distance/kernel
    // value bounds
    dsqd_range.lo = qnode->bound ().MinDistanceSq (rnode->bound ());
    dsqd_range.hi = qnode->bound ().MaxDistanceSq (rnode->bound ());
    kernel_value_range = kernel_.RangeUnnormOnSq (dsqd_range);


    // the new lower bound after incorporating new info for each dimension
    for (int i = 0; i < rset_.n_rows () + 1; i++){
      dl[i] = rnode->stat().weight_of_dimension[i] * kernel_value_range.lo;
      du[i] = -1 * rnode->stat().weight_of_dimension[i] * (1 -kernel_value_range.hi);
    }

    // refine the lower bound using the new lower bound info

    Vector new_mass_l;
    new_mass_l.Init (rset_.n_rows () + 1);
    new_mass_l.SetAll (0);

    for (int i = 0; i < rset_.n_rows () + 1; i++){  //Along each dimension

      new_mass_l[i] = stat.mass_l[i] + dl[i];
      double allowed_err = tau_ * new_mass_l[i] *
	((double) (rnode->stat().weight_of_dimension[i]) /
	 ((double) (rroot_->stat().weight_of_dimension[i])));

      // this is error per each query/reference pair for a fixed query
      double m = 0.5 * (kernel_value_range.hi - kernel_value_range.lo);

      // this is total maximumn error for each query point
      double error = m * rnode->stat ().weight_of_dimension[i];

      // check pruning condition
      if (error > allowed_err){

	dl.SetZero();
	du.SetZero();
	return 0;
      }
    }
    //could prune along every dimension. I return from here only 
    // if i am able to prune along each dimension. else i return earlier
 
    number_of_prunes++;
    return 1;
  }

  /** determine which of the node to expand first */
  void BestNodePartners (Tree * nd, Tree * nd1, Tree * nd2, Tree ** partner1,Tree ** partner2){

    double d1 = nd->bound ().MinDistanceSq (nd1->bound ());
    double d2 = nd->bound ().MinDistanceSq (nd2->bound ());

    if (d1 <= d2){
      *partner1 = nd1;
      *partner2 = nd2;
    }
    else{
      *partner1 = nd2;
      *partner2 = nd1;
    }
  }

  /** This is kind of the major function which calls other functions 
   * here is where we do a complete dual tree recursion
   */
  void FVectorEstimate (Tree * qnode, Tree * rnode){

    /** temporary variable for storing lower bound change */

    Vector dl;
    Vector du;
    dl.Init (rset_.n_rows () + 1);
    du.Init (rset_.n_rows () + 1);
    dl.SetAll (0);
    du.SetAll (0);

    // temporary variable for holding distance/kernel value bounds
    DRange dsqd_range;
    DRange kernel_value_range;

    // query node statistics
    FastVectorStat & stat = qnode->stat ();

    // left child and right child of query node statistics
    FastVectorStat *left_stat = NULL;
    FastVectorStat *right_stat = NULL;

    /** We first do updatebounds before the start of recursion
     * This is like a cleaning up task where we add the owed values stored in 
     * the parent node to the mass values. 
     */ 

    UpdateBounds (qnode, qnode->stat ().owed_l, qnode->stat ().owed_u);

    //Since owed_l and owed_u have already been incorporated flush them out
    for (int i = 0; i < stat.owed_l.length (); i++){
      stat.owed_l[i] = stat.owed_u[i] = 0;
    }

    if (!qnode->is_leaf ()){

      left_stat = &(qnode->left ()->stat ());
      right_stat = &(qnode->right ()->stat ());

    // try finite difference pruning first

    if (Prunable (qnode, rnode, dsqd_range, kernel_value_range, dl, du)){

	UpdateBounds (qnode, dl, du);
	return;
      }
    }

    //Pruning failed........

    // for leaf query node
    if (qnode->is_leaf ()){

      // for leaf pairs, go exhaustive
      if (rnode->is_leaf ()){ 

	//We have already checked for prunability so there is no need to do that check once again 

	FVectorEstimateBase (qnode, rnode);
	return;
      }

      // for non-leaf reference, expand reference node
      else{

	Tree *rnode_first = NULL, *rnode_second = NULL;
	BestNodePartners (qnode, rnode->left (), rnode->right (),
			  &rnode_first, &rnode_second);
	FVectorEstimate (qnode, rnode_first);
	FVectorEstimate (qnode, rnode_second);
	return;
      }
    }

    // for non-leaf query node
    else{
      // for a leaf reference node, expand query node
      if (rnode->is_leaf ()){

	Tree *qnode_first = NULL, *qnode_second = NULL;
	BestNodePartners (rnode, qnode->left (), qnode->right (),
			  &qnode_first, &qnode_second);
	FVectorEstimate (qnode_first, rnode);
	FVectorEstimate (qnode_second, rnode);
	//	return;
      }

      // for non-leaf reference node, expand both query and reference nodes
      else{

	Tree *rnode_first = NULL, *rnode_second = NULL;
	BestNodePartners (qnode->left (), rnode->left (), rnode->right (),
			  &rnode_first, &rnode_second);
	FVectorEstimate (qnode->left (), rnode_first);
	FVectorEstimate (qnode->left (), rnode_second);

	BestNodePartners (qnode->right (), rnode->left (),
			  rnode->right (), &rnode_first, &rnode_second);
	FVectorEstimate (qnode->right (), rnode_first);
	FVectorEstimate (qnode->right (), rnode_second);
      }
      stat.MergeChildBounds (*left_stat, *right_stat, rset_.n_rows()+ 1);
    }
  }

  /** The postprocessing step is done once the algorithm is over 
   * this is because during the run of the algorithm owed values 
   * might not have completely added to the mass-l and mass_u values
   * hence by doing a postprocessing step, we peroclate these values down 
   * the tree to the leaf nodes where they finally get added to the mass_l and 
   * mass_u values 
   */
  void PostProcess (Tree * qnode){

    FastVectorStat & stat = qnode->stat ();

    // for leaf query node
    if (qnode->is_leaf ()){

      UpdateBounds (qnode, qnode->stat ().owed_l, qnode->stat ().owed_u);

      /** Realise that we calculate the vector estimates for each query point partially by exhaustive case
       *  and partially by pruning. vector_estimate_l and vector_estimate_u hold the vector estimates due to 
       *  exhaustive calculations and more_l and more-u hold the vector estimate due to pruning. Final estimate 
       * is half the sum of all values 
       */
      for (index_t q = qnode->begin (); q < qnode->end (); q++){ //for each point
	for (int i = 0; i < rset_.n_rows () + 1; i++){	//Along each dimension
		
	  vector_estimate_e_[q][i] =
	    (vector_estimate_l_[q][i] + qnode->stat ().more_l[i] +
	     vector_estimate_u_[q][i] + qnode->stat ().more_u[i]) / 2.0;
	}
      }
    }
    else{
      //It is a non-leaf node
      UpdateBounds (qnode, stat.owed_l, stat.owed_u);
      PostProcess (qnode->left ());
      PostProcess (qnode->right ());
    }
  }

  /** We diont use this function, but we can use them to maintain numerical stability */
  void NormalizeVector_Estimate (Tree * rnode){

    index_t norm_const=1;
    for (index_t q = 0; q < qset_.n_cols (); q++){
      for (index_t i = 0; i < rset_.n_rows () + 1; i++){	//this is the dimension
	vector_estimate_l_[q][i] /= norm_const;
	vector_estimate_e_[q][i] /= norm_const;
	vector_estimate_u_[q][i] /= norm_const;
      }
    }
  }

 public:

  // constructor/destructor
  FastVectorCalculation (){
  }

  ~FastVectorCalculation (){
    //delete qroot_;
    //delete rroot_;
      
  }

  // getters and setters

  /**get the reference weights*/

  Vector get_reference_weights (){

    return rset_weights_;
  }

  /** Get the perumutation*/

  ArrayList<int>& get_new_from_old_r(){
    
    return new_from_old_r_;
  }

  ArrayList<int>& get_old_from_new_r(){
    
    return old_from_new_r_;
  }
  /** get the reference dataset */
  Matrix & get_reference_dataset (){

    return rset_;
  }

  /** get the query dataset */
  Matrix & get_query_dataset (){

    return qset_;
  }

  /** get the vector estimate */
 double get_vector_estimates (int q,int d){

    return vector_estimate_e_[q][d];
  }

  // interesting functions.......

 /** This function sets the value of mass_u of each query node to the maxzmium value that the
  *  vector B^TWy can take
  */
  void SetUpperBoundOfVector (Tree * qnode){

    if (qnode->is_leaf ()){

      qnode->stat ().mass_u.Init (rset_.n_rows () + 1);
      //Initialize the value
      for (int i = 0; i < rset_.n_rows () + 1; i++){

	qnode->stat ().mass_u[i] = qroot_->stat ().weight_of_dimension[i];
      }
      return;
    }

    //initialize the size
    qnode->stat ().mass_u.Init (rset_.n_rows () + 1);
 
    for (int i = 0; i < rset_.n_rows () + 1; i++){

      qnode->stat ().mass_u[i] = qroot_->stat ().weight_of_dimension[i];
    }
    SetUpperBoundOfVector(qnode->left ());
    SetUpperBoundOfVector (qnode->right ());
  }


  void Compute (double tau){

    // set accuracy parameter
    tau_ = tau;

    //PreProcess first.....

    PreProcess (qroot_);
    PreProcess (rroot_);

    //printf ("Preprocessing complete....\n");

    //Initialize all the vector_estimate properly

    vector_estimate_l_.Init (qset_.n_cols ());
    vector_estimate_u_.Init (qset_.n_cols ());
    vector_estimate_e_.Init (qset_.n_cols ());

    for (int i = 0; i < qset_.n_cols (); i++){

      vector_estimate_l_[i].Init (rset_.n_rows () + 1);	//size= number of dimensions+1
      vector_estimate_l_[i].SetAll (0);

      vector_estimate_e_[i].Init (rset_.n_rows () + 1);
      vector_estimate_e_[i].SetAll (0);
    }

    for (int i = 0; i < qset_.n_cols (); i++){	//for every point

      vector_estimate_u_[i].Init (rset_.n_rows () + 1);

      for (int j = 0; j < rset_.n_rows () + 1; j++){	//for every dimension

	vector_estimate_u_[i][j] = rroot_->stat ().weight_of_dimension[j];
      }
    }
    SetUpperBoundOfVector(qroot_);

    // call main routine
   
    FVectorEstimate (qroot_, rroot_);

    // postprocessing step for finalizing the sums
    PostProcess (qroot_);

  
  }


  void Init (){			//This is the Init function of FastVectorCalculation
    
    Dataset ref_dataset;

    // read in the number of points owned by a leaf
    int leaflen = fx_param_int (NULL, "leaflen", 2);

    // read the datasets
    const char *rfname = fx_param_str_req (NULL, "data");
    const char *qfname = fx_param_str (NULL, "query", rfname);


    // read reference dataset
    ref_dataset.InitFromFile (rfname);
    rset_.Own (&(ref_dataset.matrix ()));

    // read the reference weights.
    char *rwfname = NULL;

    if (fx_param_exists (NULL, "dwgts")){

      //rwfname is the filename having the refernece weights
      rwfname = (char *) fx_param_str (NULL, "dwgts", NULL);
    }

    if (rwfname != NULL){

      Dataset ref_weights;
      ref_weights.InitFromFile (rwfname);
      rset_weights_.Copy (ref_weights.matrix ().GetColumnPtr (0), ref_weights.matrix ().n_rows ());	//Note rset_weights_ is a vector of weights
     
    }

    else{

      rset_weights_.Init (rset_.n_cols ());
      rset_weights_.SetAll (1);
     
    }

    printf("the weights are ..\n");
    // rset_weights_.PrintDebug();
   
    if (!strcmp (qfname, rfname)){

      qset_.Alias (rset_);
    }
    else{

      Dataset query_dataset;
      query_dataset.InitFromFile (qfname);
      qset_.Own (&(query_dataset.matrix ()));
    }

    // scale dataset if the user wants to. NOTE: THIS HAS TO BE VERIFIED.....

    if (!strcmp (fx_param_str (NULL, "scaling", NULL), "range")){

      scale_data_by_minmax ();
      printf("scaling over and will exit now..\n");
      
    }

    // construct query and reference trees. This also fills up the statistics in the reference tree


    printf("the refernece file is ..\n");
    printf("sixe of ref file is %d\n",rset_.n_cols()*rset_.n_rows());

    printf("the QUery file is ..\n");
    printf("sixe of query file is %d\n",qset_.n_cols()*qset_.n_rows());


    
    fx_timer_start (NULL, "tree_first");
    rroot_ = tree::MakeKdTreeMidpoint < Tree > (rset_, leaflen, &old_from_new_r_, &new_from_old_r_);
    qroot_=tree::MakeKdTreeMidpoint < Tree > (qset_, leaflen, NULL, NULL);
  
    fx_timer_stop (NULL, "tree_first");

    // initialize the kernel

    kernel_.Init (fx_param_double_req (NULL, "bandwidth"));
    fast_kde_base=0;
    number_of_prunes=0;
  }


  /*void PrintDebug (){

    //   FILE *stream = stdout;
    const char *fname = NULL;
    FILE *fp;

    if ((fname = fx_param_str (NULL, "fast_kde_output", NULL)) != NULL){


      fp = fopen (fname, "w+");
    }
    for (index_t q = 0; q < qset_.n_cols (); q++){

      fprintf(fp,"Point:");
      for(index_t z=0;z<qset_.n_rows();z++){

	fprintf(fp,"%f",qset_.get(z,q));
	fprintf(fp," ");
      }
      fprintf(fp,"\n");

      for (index_t d = 0; d < rset_.n_rows () + 1; d++){

	fprintf (fp, "%f\n", densities_e_[q][d]);
      }
    }

    fclose (fp);
    }*/

};

#endif
