#ifndef KDE_H
#define KDE_H
#define MAXDOUBLE 32768.0
#include "fastlib/fastlib_int.h"


template < typename TKernel > class NaiveKde{

 private:

  /** query dataset */
  Matrix qset_;

  /** reference dataset */
  Matrix rset_;

  /** kernel */
  TKernel kernel_;

  /** computed densities */
  ArrayList < Vector > densities_;

  /**Reference weights*/
  Vector rset_weights_;

 public:
  void Compute (){

    printf ("\nStarting naive KDE computations...\n");
    fx_timer_start (NULL, "naive_kde_compute");

    // compute unnormalized sum
    for (index_t q = 0; q < qset_.n_cols (); q++){	//for each query point
	
      const double *q_col = qset_.GetColumnPtr (q);
      for (index_t d = 0; d < rset_.n_rows () + 1; d++){	//along each direction
	for (index_t r = 0; r < rset_.n_cols (); r++){	//for each reference point
	  if (d == 0){
	   
	    const double *r_col = rset_.GetColumnPtr (r);
	    double dsqd =la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	    double val = kernel_.EvalUnnormOnSq (dsqd);
	    val *= rset_weights_[r];
	    densities_[q][d] += val;
	  }
	  else{

	    const double *r_col = rset_.GetColumnPtr (r);
	    double dsqd =
	      la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	    double val = kernel_.EvalUnnormOnSq (dsqd);
	    val *= rset_weights_[r];
	    val *= rset_.get (d - 1, r);
	    densities_[q][d] += val;
	  }
	}
      }
    }

    //Density estimates from naive calculations.....
    for(int i=0;i<qset_.n_cols();i++){
      for(int j=0;j<rset_.n_rows()+1;j++){
	printf("density from naive calculation is %f\n",densities_[i][j]);
      }
    }
    // then normalize it
    /*    for (index_t q = 0; q < qset_.n_cols (); q++){	//for each query point
	  for (index_t d = 0; d < rset_.n_rows () + 1; d++){	//along each dimension
	  densities_[q][d] /= (rset_.n_cols());
      }
      }*/
  }

  void Init (){

    densities_.Init (qset_.n_cols ());
    for (index_t i = 0; i < qset_.n_cols (); i++){
      densities_[i].Init (rset_.n_rows () + 1);
      densities_[i].SetAll (0);
    }
  }

  void Init (Matrix & qset, Matrix & rset){
    
    // get datasets
    qset_.Alias (qset);
    rset_.Alias (rset);
    
    // get bandwidth
    kernel_.Init (fx_param_double_req (NULL, "bandwidth"));
   

    densities_.Init (qset_.n_cols ());
    //printf ("density initialized as a 1-d array..\n");
    for (index_t i = 0; i < qset_.n_cols (); i++){
      densities_[i].Init (rset_.n_rows () + 1);
      densities_[i].SetZero ();
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

  void PrintDebug (){

    FILE *fp;
    const char *fname;

    if (fx_param_exists (NULL, "naive_kde_output")){

      fname = fx_param_str (NULL, "naive_kde_output", NULL);
      fp = fopen (fname, "w+");
    }
    for (index_t q = 0; q < qset_.n_cols (); q++){

      for (index_t d = 0; d < rset_.n_rows () + 1; d++){
	fprintf (fp, "%f\n", densities_[q][d]);
      }
    }
  }

  void ComputeMaximumRelativeError (ArrayList < Vector > density_estimate) {

    printf ("Came to compute maximum relative error...\n");
    double max_rel_err = 0;
    for (index_t q = 0; q < qset_.n_cols (); q++){
      for (index_t d = 0; d < rset_.n_rows () + 1; d++){

	printf ("The values being compared are ...\n");
	printf ("density estimate is %f and density is %f\n",
		density_estimate[q][d], densities_[q][d]);
	double rel_err =
	  fabs (density_estimate[q][d] -
		densities_[q][d]) / densities_[q][d];

	if (rel_err > max_rel_err){
	  max_rel_err = rel_err;
	}
      }
    }

    fx_format_result (NULL, "maxium_relative_error_for_fast_KDE", "%g",
		      max_rel_err);
  }

};  //Class NaiveKDE ends........

/*****************************************************************************************************/

template < typename TKernel > class FastKde{
 public:

  // forward declaration of KdeStat class
  class KdeStat;

  // our tree type using the KdeStat
  typedef BinarySpaceTree < DHrectBound < 2 >, Matrix, KdeStat > Tree;

  class KdeStat{

  public:

    /** This is the weights dimension wise. $\sum (x_{ij}y_{i})$. It is a D+1 dimensional array */

    Vector weight_of_dimension;

    /** lower bound on the densities for the query points owned by this node . Realise that now it is a 1-D vector
     */
    Vector mass_l;

    /**
     * additional offset for the lower bound on the densities for the query
     * points owned by this node (for leaf nodes only). Realise that now it is a 1-D vector
     */
    Vector more_l;

    /**
     * lower bound offset passed from above. Realise that now it is a 1-D vector
     */
    Vector owed_l;

    /** upper bound on the densities for the query points owned by this node. Realise that now it is a 1-D vector
     */
    Vector mass_u;

    /**
     * additional offset for the upper bound on the densities for the query
     * points owned by this node (for leaf nodes only). Realise that now it is a 1-D vector
     */
    Vector more_u;

    /**
     * upper bound offset passed from above. Realise that now it is a 1-D vector
     */
    Vector owed_u;


    //These are the Init functions of KdeStat
    void Init (){

    }

    void Init (const Matrix & dataset, index_t & start, index_t & count){

      Init ();
    }

    void Init (const Matrix & dataset, index_t & start, index_t & count,
	       const KdeStat & left_stat, const KdeStat & right_stat){
      Init ();
    }

    void MergeChildBounds (KdeStat & left_stat, KdeStat & right_stat,
			   index_t dim){
      // improve lower and upper bound
      for (index_t i = 0; i < dim; i++)
	{
	  mass_l[i] =
	    max (mass_l[i], min (left_stat.mass_l[i], right_stat.mass_l[i]));
	  mass_u[i] =
	    min (mass_u[i], max (left_stat.mass_u[i], right_stat.mass_u[i]));
	}
    }

    KdeStat (){
    }

    ~KdeStat (){
    }

  };				//WITH THIS CLASS KDESTAT COMES TO AN END

  //Private member of FastKde
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

  /** lower bound on the densities.This will now be a 2-D vector*/
  ArrayList <Vector> densities_l_;

  /** densities computed .This will now be a 2-D vector*/
  ArrayList <Vector> densities_e_;

  /** upper bound on the densities.This will now be a 2-D vector */
  ArrayList <Vector> densities_u_;

  /** accuracy parameter */
  double tau_;

  /**Number of base case operations*/
  int fast_kde_base;
 
  // preprocessing: scaling the dataset; this has to be moved to the dataset
  // module
  /* scales each attribute to 0-1 using the min/max values */

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
      
      printf ("Dimension %d range: [%g, %g]\n", i, min_coord, max_coord);
      
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
  
  /** Initialize the statistics */
  
  void PreProcess (Tree * node){
    
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
	      rset_.get (i - 1, j) * rset_weights_[j];
	  }
	  else{
	    node->stat ().weight_of_dimension[i] += rset_weights_[j];
	    
	  }
	}
      }
      printf ("THE WEIGHT VECTOR I WILL BE RETURNING IS ...\n");
      for (int i = 0; i < rset_.n_rows () + 1; i++){
	printf ("weight is %f\n", node->stat ().weight_of_dimension[i]);
      }
    }
      
    
    // for non-leaf node, recurse
    else{
      //printf ("This is not a leaf node...\n");
      
      PreProcess (node->left ());
      PreProcess (node->right ());

      for (int count = 0; count < rset_.n_rows () + 1; count++){
	
	node->stat ().weight_of_dimension[count] =
	  node->left ()->stat ().weight_of_dimension[count] +
	  node->right ()->stat ().weight_of_dimension[count];
      }
      
    }
    
  }

  void UpdateBounds (Tree * qnode, Vector dl, Vector du){

    // query self statistics
    KdeStat & qstat = qnode->stat ();
    
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
	qstat.more_l[t] += dl[t];	//Why are we doing this. Why dowe maintain a field called qstat.more_l_ for a leaf node
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
    //Since owed_l and owed_u have already been incorporated into 
    for (int i = 0; i < qstat.owed_l.length (); i++){
      qstat.owed_l[i] = qstat.owed_u[i] = 0;
    }
  }

  /** exhaustive base KDE case */
  void FKdeBase (Tree * qnode, Tree * rnode){
    printf ("In FKdeBase ...\n");

    fast_kde_base++;
    printf("with this fast_kde_base is %d\n",fast_kde_base);

    //subtract because now you are doing exhaustive computation

    for(index_t d=0;d<rset_.n_rows()+1;d++){ //along each dimension
      printf("Before subtraction i have more_u is %f\n",qnode->stat().more_u[d]);
      qnode->stat().more_u[d]-=rnode->stat().weight_of_dimension[d];
      printf("After subtraction i have more_u is %f\n",qnode->stat().more_u[d]);
    }
     
    // compute unnormalized sum
    for (index_t q = qnode->begin (); q < qnode->end (); q++){

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
	    densities_l_[q][i] +=
	      ker_value * rset_weights_[r] * rset_.get (i - 1, r);
	    densities_u_[q][i] +=
	      ker_value * rset_weights_[r] * rset_.get (i - 1, r);
	  }
	  else{
	    densities_l_[q][i] += ker_value * rset_weights_[r];
	    densities_u_[q][i] += ker_value * rset_weights_[r];

	  }
	}
      }
    }

    // get a tighter lower and upper bound for every dimension by looping over each query point
    // in the current query leaf node

    Vector min_l;
    min_l.Init (rset_.n_rows () + 1);

    Vector max_u;
    max_u.Init (rset_.n_rows () + 1);
    min_l.SetAll (MAXDOUBLE);
    max_u.SetAll (-MAXDOUBLE);

    for (index_t t = 0; t < rset_.n_rows () + 1; t++){
      for (index_t q = qnode->begin (); q < qnode->end (); q++){
	if (densities_l_[q][t] < min_l[t]){

	  min_l[t] = densities_l_[q][t];
	}
	if (densities_u_[q][t] > max_u[t]){
	  max_u[t] = densities_u_[q][t];
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
   * checking for prunability of the query and the reference pair using
   * four types of pruning methods
   */

  /** checking for prunability of the query and the reference pair */
  int Prunable (Tree * qnode, Tree * rnode, DRange & dsqd_range,
		DRange & kernel_value_range, Vector dl, Vector du){

    // query node stat
    KdeStat & stat = qnode->stat ();
    // printf("The error tolerance is %f\n",tau_);


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
	if(qnode->is_leaf()&&rnode->is_leaf()){
	  printf("Both are leaf nodes and they didnt get pruned..\n");
	}
	dl.SetZero();
	du.SetZero();
	return 0;
      }
    }
    //could prune along every dimension
    if(qnode->is_leaf()&&rnode->is_leaf()){
      printf("both are leaf nodes and pruning took place..\n");
    }
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

  /** canonical fast KDE case */
  void FKde (Tree * qnode, Tree * rnode){

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
    KdeStat & stat = qnode->stat ();

    // left child and right child of query node statistics
    KdeStat *left_stat = NULL;
    KdeStat *right_stat = NULL;
    UpdateBounds (qnode, qnode->stat ().owed_l, qnode->stat ().owed_u);

    if (!qnode->is_leaf ()){

      left_stat = &(qnode->left ()->stat ());
      right_stat = &(qnode->right ()->stat ());
      // stat.MergeChildBounds (*left_stat, *right_stat, rset_.n_rows()+ 1); //I think this should not be here
    }

    // try finite difference pruning first
    if (Prunable (qnode, rnode, dsqd_range, kernel_value_range, dl, du)){

	UpdateBounds (qnode, dl, du);
	return;
      }

    //Pruning failed........

    // for leaf query node
    if (qnode->is_leaf ()){

      // for leaf pairs, go exhaustive
      if (rnode->is_leaf ()){

	FKdeBase (qnode, rnode);
	return;
      }

      // for non-leaf reference, expand reference node
      else{

	Tree *rnode_first = NULL, *rnode_second = NULL;
	BestNodePartners (qnode, rnode->left (), rnode->right (),
			  &rnode_first, &rnode_second);
	FKde (qnode, rnode_first);
	FKde (qnode, rnode_second);
	return;
      }
    }

    // for non-leaf query node
    else{
      // for a leaf reference node, expand query node
      if (rnode->is_leaf ()){

	Tree *qnode_first = NULL, *qnode_second = NULL;

	//stat.PushDownTokens(*left_stat, *right_stat, NULL, NULL,&stat.mass_t_);
	BestNodePartners (rnode, qnode->left (), qnode->right (),
			  &qnode_first, &qnode_second);
	FKde (qnode_first, rnode);
	FKde (qnode_second, rnode);
	//	return;
      }

      // for non-leaf reference node, expand both query and reference nodes
      else{

	Tree *rnode_first = NULL, *rnode_second = NULL;
	//stat.PushDownTokens(*left_stat, *right_stat, NULL, NULL,&stat.mass_t_);

	BestNodePartners (qnode->left (), rnode->left (), rnode->right (),
			  &rnode_first, &rnode_second);
	FKde (qnode->left (), rnode_first);
	FKde (qnode->left (), rnode_second);

	BestNodePartners (qnode->right (), rnode->left (),
			  rnode->right (), &rnode_first, &rnode_second);
	FKde (qnode->right (), rnode_first);
	FKde (qnode->right (), rnode_second);
	//	return;
      }
      stat.MergeChildBounds (*left_stat, *right_stat, rset_.n_rows()+ 1);
    }
  }

  /** post processing step */
  void PostProcess (Tree * qnode){

    KdeStat & stat = qnode->stat ();

    // for leaf query node
    if (qnode->is_leaf ()){

      UpdateBounds (qnode, qnode->stat ().owed_l, qnode->stat ().owed_u);

      for (index_t q = qnode->begin (); q < qnode->end (); q++){ //for each point
	for (int i = 0; i < rset_.n_rows () + 1; i++){	//Along each dimension
		
	  printf("We have the following estimates....\n");
	  printf("density_lower is %f\n",densities_l_[q][i]);
	  printf("density upper estimate is %f\n",densities_u_[q][i]);
	  printf("lower estimate of more is %f\n",qnode->stat().more_l[i]);
	  printf("Upper estimate of more is %f\n",qnode->stat().more_u[i]);
	  densities_e_[q][i] =
	    (densities_l_[q][i] + qnode->stat ().more_l[i] +
	     densities_u_[q][i] + qnode->stat ().more_u[i]) / 2.0;
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

  void NormalizeDensities (Tree * rnode){

    index_t norm_const=1;
    for (index_t q = 0; q < qset_.n_cols (); q++){
      for (index_t i = 0; i < rset_.n_rows () + 1; i++){	//this is the dimension
	densities_l_[q][i] /= norm_const;
	densities_e_[q][i] /= norm_const;
	densities_u_[q][i] /= norm_const;
      }
    }
  }

 public:

  // constructor/destructor
  FastKde (){
  }

  ~FastKde (){

    printf ("Have come to destructor of fastkde...\n");
    //delete qroot_;
    //delete rroot_;
      
  }

  // getters and setters

  void set_total_sum_of_weights (Vector rset_weights_){
    double total_sum_of_weights_ = 0;
    for (int i = 0; i < rset_weights_.length (); i++){
      total_sum_of_weights_ += rset_weights_[i];
    }
  }

  /*void set_total_sum_of_weights (index_t value){

  total_sum_of_weights_ = value;
  }*/
  /**get the reference weights*/

  Vector get_reference_weights (){

    return rset_weights_;
  }

  /** get the reference dataset */
  Matrix & get_reference_dataset (){

    return rset_;
  }

  /** get the query dataset */
  Matrix & get_query_dataset (){

    return qset_;
  }

  /** get the density estimate */
  const Vector & get_density_estimates (int i){

    return densities_e_[i];
  }

  // interesting functions.......


  void SetUpperBoundOfDensity (Tree * qnode){

    if (qnode->is_leaf ()){

      qnode->stat ().mass_u.Init (rset_.n_rows () + 1);
      //Initialize the value
      for (int i = 0; i < rset_.n_rows () + 1; i++){

	qnode->stat ().mass_u[i] = qnode->stat ().weight_of_dimension[i];
      }
      return;
    }

    //initialize the size
    qnode->stat ().mass_u.Init (rset_.n_rows () + 1);
    //Initialize the value
    for (int i = 0; i < rset_.n_rows () + 1; i++){

      qnode->stat ().mass_u[i] = qnode->stat ().weight_of_dimension[i];
    }
    SetUpperBoundOfDensity (qnode->left ());
    SetUpperBoundOfDensity (qnode->right ());
  }


  void Compute (double tau){

    printf ("Came to compute function...\n");
    // set accuracy parameter
    tau_ = tau;

    //num_finite_difference_prunes_ = num_farfield_to_local_prunes_ =num_farfield_prunes_ = num_local_prunes_ = 0;

    // fx_timer_start(NULL, "fast_kde_compute");

    //printf ("Will  call FKde....\n");
    //printf ("Will preprocess now....\n");
   

    //PreProcess first.....
    PreProcess (qroot_);
    //printf ("done with preprocessing of the query tree.....\n");

    PreProcess (rroot_);
    //printf ("Preprocessing complete....\n");

    //Initialize all the densities properly

    densities_l_.Init (qset_.n_cols ());
    densities_u_.Init (qset_.n_cols ());
    densities_e_.Init (qset_.n_cols ());

    for (int i = 0; i < qset_.n_cols (); i++){

      densities_l_[i].Init (rset_.n_rows () + 1);	//size= number of dimensions+1
      densities_l_[i].SetAll (0);

      densities_e_[i].Init (rset_.n_rows () + 1);
      densities_e_[i].SetAll (0);
    }

    for (int i = 0; i < qset_.n_cols (); i++){	//for every point

      densities_u_[i].Init (rset_.n_rows () + 1);

      for (int j = 0; j < rset_.n_rows () + 1; j++){	//for every dimension

	densities_u_[i][j] = rroot_->stat ().weight_of_dimension[j];
      }
    }
    SetUpperBoundOfDensity (qroot_);

    // call main routine
    FKde (qroot_, rroot_);

    // postprocessing step for finalizing the sums
    PostProcess (qroot_);

    /**normalize densities*/

    // NormalizeDensities (rroot_);
  }


  void Init (){			//This is the Init function of FastKde
    
    Dataset ref_dataset;

    // read in the number of points owned by a leaf
    int leaflen = fx_param_int (NULL, "leaflen", 1);

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
      //set_total_sum_of_weights (rset_weights_);
    }

    else{

      rset_weights_.Init (rset_.n_cols ());
      rset_weights_.SetAll (1);
      //set_total_sum_of_weights (rset_.n_cols ());
    }
   
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
    }

    // construct query and reference trees. This also fills up the statistics in the reference tree

    fx_timer_start (NULL, "tree_d");
    rroot_ = tree::MakeKdTreeMidpoint < Tree > (rset_, leaflen);
    qroot_ = tree::MakeKdTreeMidpoint < Tree > (qset_, leaflen);
    fx_timer_stop (NULL, "tree_d");

    // initialize the kernel

    kernel_.Init (fx_param_double_req (NULL, "bandwidth"));
    printf("In the fast kde module....\n");
    fast_kde_base=0;
    number_of_prunes=0;
  }


  void PrintDebug (){

    //   FILE *stream = stdout;
    const char *fname = NULL;
    FILE *fp;

    if ((fname = fx_param_str (NULL, "fast_kde_output", NULL)) != NULL){


      fp = fopen (fname, "w+");
    }
    for (index_t q = 0; q < qset_.n_cols (); q++){

      for (index_t d = 0; d < rset_.n_rows () + 1; d++){

	fprintf (fp, "%f\n", densities_e_[q][d]);
      }
    }
    printf("Total number of base calulations are %d\n",fast_kde_base);
    printf("Number of prunes are...%d\n",number_of_prunes);
    fclose (fp);
  }

};

#endif
