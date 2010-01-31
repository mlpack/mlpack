/* THIS FILE COMPUTES B^TWB by pruning the entire matrix */


#ifndef REGRESSION_NEW1_H
#define REGRESSION_NEW1_H
#define leaflen 1
#include <values.h>


/** This is a friend function and will be used by both the classes defined in this file*/

// This function will calculate the squared frobenius norm of a matrix. 
// It does so by caculating the sum of the squares of all elements

 double SquaredFrobeniusNorm(Matrix &a){
   
   double sqd_frobenius_norm=0;
   for(int rows=0;rows<a.n_rows();rows++){
     
     for(int cols=0;cols<a.n_cols();cols++){
       sqd_frobenius_norm+=a.get(rows,cols)*a.get(rows,cols);
     }
   }
   return sqd_frobenius_norm;
 }

/** 
 * We now define a class called NaiveRegression_new1. 
 * This class will enable us to compute both the B^TWB and B^TWY matrix naively. 
 * We shall later compare the performance of the naive method with that of our
 * faster algorithm which employs a dual tree framework
*/

template <typename TKernel> class NaiveRegression_new1{

 private:

  /** query dataset */
  Matrix qset_;

  /** reference dataset */
  Matrix rset_;

  /** kernel */
  TKernel kernel_;

  /** The ArrayList of matrices  stores the Matrix B^TWB which has been computed naively for each query point**/

  ArrayList <Matrix> results_;

 public:

  /**Interesting functions................. */

  /** This is a friend function and will be used by both the naive class and the fast class 
   * Hence it has been defined outside the class
   */

  double friend SquaredFrobeniusNorm(Matrix &a);                   

  /** This compares the results obtained from the fast calculations with the calculations of naive method */

  void CompareWithFast(ArrayList<Matrix> &fast){

    printf("Came to compare with fast..\n");
    FILE *fp;
    fp=fopen("squared_frobenius_error.txt","w+");

    FILE *gp;
    gp=fopen("high_relative_error.txt","w+");

    // Lets define some temporary variables for scratchpad calculations 
    double sqdfast,sqdnaive,relative_error,max_error=0.0;
    double average_relative_error=0;
    index_t number_of_violators=0;

    for(index_t i=0;i<qset_.n_cols();i++){

      //sqdfast hold the squared frobenius norm of the B^TWB matrix caclulated for the query point i by naive method

      sqdfast=SquaredFrobeniusNorm(fast[i]);

      //Similarily sqdnaive  hold the squared frobenius norm of the B^TWB matrix caclulated for the query point i 
      // Calculated by fast method

      sqdnaive=SquaredFrobeniusNorm(results_[i]);
 
      //Now calculate the squared frobenius norm of the 
      //difference between the B^TWB matrix for the query point i 
      //calculated by fast methods and by naive methods.temp <- fast[i]-naive[i]

      Matrix temp;
      la::SubInit (results_[i], fast[i], &temp);
      relative_error=fabs(SquaredFrobeniusNorm(temp))/sqdnaive;
      average_relative_error+=relative_error;
      fprintf(fp,"relative error:%f\n",relative_error);
      if(max_error<=relative_error){
	
	max_error=relative_error;
      }

      if(relative_error>0){

	fprintf(gp,"High relative error...\n");
	fprintf(gp,"relative_error is %f\n",relative_error);
	number_of_violators++;
      }
     
    }
    
    fprintf(fp,"Average Relative error was %f\n",average_relative_error);
    fprintf(fp,"Maximum error was %f\n",max_error);
    printf("average relative error was %f\n",average_relative_error);
    printf("Maximum reltive error is %f\n",max_error);
    // printf("Number of violations are %d\n",number_of_violators);
    
  }

  void Compute (){
    
    //We shall now evaluate B^TBW for each query point. Note because W is a diagonal matrix.
     
    for(index_t q=0;q<qset_.n_cols();q++){
      
      const double *q_col = qset_.GetColumnPtr (q); //get the query point
      
      for(index_t r=0;r<rset_.n_cols();r++){
	
	//B_t is the transpose of the matrix B
	Matrix B_t;
	B_t.Init(rset_.n_rows()+1,1);
	
	/*Form B^T matrix */
	for(index_t row=0;row<rset_.n_rows()+1;row++){
	  
	  if(row==0){

	    B_t.set(row,0,1);
	  }

	  else
	    {

	      B_t.set(row,0,rset_.get(row-1,r));
	    }
	}

   	const double *r_col = qset_.GetColumnPtr (r); //get the reference point	

	//Calculate the squared distance between the query point 
	// and the reference point. then calculate the kernel value

	double dsqd =la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	double val = kernel_.EvalUnnormOnSq (dsqd);

	/* temp2 holds B */
	Matrix temp2; 
	la::TransposeInit(B_t,&temp2); 

	//temp3 holds B^TB
	Matrix temp3;
	la:: MulInit(B_t,temp2,&temp3); 

	// scale B^TB by val 
	// temp3 holds B^TB

	la::Scale(val,&temp3);
	la::AddTo(temp3,&results_[q]);
      }
    }
  }

  /** This function initializes the paramters to be used by the class NaiveRegression_new1 */
    
  void Init (Matrix query_dataset, Matrix reference_dataset){

    //First copy the datasets
    qset_.Copy(query_dataset);
    rset_.Copy(reference_dataset);
    
    //Initialize the kernel
    kernel_.Init (fx_param_double_req (NULL, "bandwidth"));

    //Initialize results_
    results_.Init (qset_.n_cols ());
    
    for (index_t i = 0; i < qset_.n_cols (); i++){
      
      results_[i].Init(rset_.n_rows()+1,rset_.n_rows()+1); //A square matrix
      results_[i].SetAll(0);
    }
    
  }
};  //Naive regression1 is done


/* This is the class which will take a dual tree approach to compute 
 * B^TWB matrix by dual tree methods
 * 
*/

template <typename TKernel> class Regression_new1{

 public:

  /**forward declaration of Regression1Stat class
   * This class holds the statistics of the tree kd-trees which are built 
   * We forward declare it so as to use in tree declaration
   */

  class Regression1Stat;

  /**Our binary tree uses Regression1Stat */

  typedef BinarySpaceTree < DHrectBound <2>, Matrix, Regression1Stat > Tree;

  /** The definition of the statistics class */

  class Regression1Stat{

  public: 

    /*The B^TB matrix. This is required to be stored in the reference tree*/

    Matrix b_tb;

    /** lower bound on the B^TWB matrix for the query points owned by this node 
     * This is useful only for the query tree
     */

    Matrix b_twb_mass_l;
    
    /**
     * additional offset for the lower bound on the B^TWB for the query
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


    //Init functions of Regression1Stat

    void Init (){

    }

    void Init (const Matrix & dataset, index_t & start, index_t & count){

      Init ();
    }

   void Init (const Matrix & dataset, index_t & start, index_t & count, const Regression1Stat & left_stat, const Regression1Stat & right_stat){ 
      Init ();
    }

    Regression1Stat (){
      
      
    }
    
    ~Regression1Stat (){
      
    }
  
 
  };//WITH THIS CLASS Regression1STAT COMES TO AN END......
  
  /*Private members of Regression_new1*/

 private:
  
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
  
  ArrayList <Matrix> b_twb_l_;
  
  /* Estimate for the B^TWB matrix for each query point*/
  ArrayList <Matrix> b_twb_e_;
  
  /* upper bound on the B^TWB matrix*/
  ArrayList <Matrix> b_twb_u_;
   
  /** accuracy parameter */
  double tau_;

  /** Regression values for the reference points */
  Vector rset_weights_;

  /** Mappings from old dataset to new dataset 
   * Remember that when we build the query tree and the
   * reference tree out of the query and reference 
   * datasets, the datasets get permuted. These 
   * arrays give a bidirectional mapping.
  */

  ArrayList <int> old_from_new_r_;
  ArrayList <int> new_from_old_r_;

  //Variables to hold the number of prunes which succeeded and 
  //number of prunes which failed

  int prunes;
  int no_prunes;

  /* Interesting functions */

  /*This is a friend function which will be used by both the classes defined in the file */

  double friend SquaredFrobeniusNorm(Matrix &a);

  //Scale the dataset by min-max method
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

  // So lets preprocess the tree. We shall allocate memory for  all the variables
  // defined in the statistic of the tree and initialize them to proper values
  void PreProcess(Tree *node){
    
    int num_of_dimensions=rset_.n_rows();

    /*First initalize the matrices*/

    /* Lets first allocate memory to all those variables 
     * which appear in the statistic of the tree 
     */

    node->stat().b_tb.Init(num_of_dimensions+1,num_of_dimensions+1);
    node->stat().b_twb_mass_l.Init(num_of_dimensions+1,num_of_dimensions+1);
    node->stat().b_twb_mass_u.Init(num_of_dimensions+1,num_of_dimensions+1);

    node->stat().b_twb_owed_l.Init(num_of_dimensions+1,num_of_dimensions+1);
    node->stat().b_twb_owed_u.Init(num_of_dimensions+1,num_of_dimensions+1);


    /**all lower bound matices will be all 0's **/

    node->stat().b_twb_mass_l.SetAll(0);
    node->stat().b_twb_owed_l.SetAll(0);


    /**Set the upper bound matrix owed_u to 0. The upper bound matrix mass_u will be set later*/

    node->stat().b_twb_owed_u.SetAll(0);


    /**Base  Case*/

    if(node->is_leaf()){


      /* Note the leaf node has 2 more variables defined in its statistic 
       * namely node->stat().b_twb_more_l and node->stat().b_twb_more_u
       */
      
      
      node->stat().b_twb_more_l.Init(num_of_dimensions+1,num_of_dimensions+1);
      node->stat().b_twb_more_u.Init(num_of_dimensions+1,num_of_dimensions+1);

 

      /* Initialize these matrices to 0 */

      node->stat().b_twb_more_l.SetAll(0);
      node->stat().b_twb_more_u.SetAll(0);

      /**Fill the B^tB matrix.
        *First form the B^T matrix 
	*The B^T matrix contains the first row full of 1's 
        *and all other rows filled with the different reference points 
	*in a column major format 
        */

    
      /** Note an interesting thing to observe is that the 
       * the preprocess step doesnt differentiate between the tree 
       * built out of the reference tree and the query tree. hence 
       * when the preprocess is called on the query tree the points in the 
       * query nodes are used to form the B^T matrix. However we only use the 
       * B^TB matrix of the reference tree
      */
      
      /** This step marks the points required to form the B^T matrix */
      
      index_t span=node->end()-node->begin();

      // Now I form a matrix called temp which will hold 
      // all points in the reference set from the index node->start() to the index node->end()

      Matrix temp;
      rset_.MakeColumnSlice(node->begin(),span,&temp);

      Matrix B_t;
      B_t.Init(rset_.n_rows()+1,node->end()-node->begin());

      //Now I form the B^T matrix from temp. The B^t matrix has first rows filled with 1's and rest all rows are
      // just the same elements as in temp. This is clear from the lines that follow

      for(index_t row=0;row<rset_.n_rows()+1;row++){

	for(index_t col=0;col<span;col++){

	  if(row==0){
	    /*Set the first row to to all 1's */

	    B_t.set(row,col,1);
	  }
	  else
	    {
	      B_t.set(row,col,temp.get(row-1,col));
	    }
	}
      }

      Matrix temp2; //temp2 hold B
      la::TransposeInit(B_t,&temp2);
      la:: MulOverwrite(B_t,temp2,&(node->stat().b_tb));
    }
     
    else{
      /* for non-leaf recurse */
      // Note B^TB_parent= B^TB_left+B^TB_right
 
      PreProcess(node->left());
      PreProcess(node->right());
      la::AddOverwrite (node->left()->stat().b_tb, node->right()->stat().b_tb, &(node->stat().b_tb));
    }
  }


  /**This is a very simple function that  tries  to find the best partner for 
    *recursion. This is done by finding out the partner that is closest to the given node(nd)  
  */

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

  /** this function updates the values of b_twb_mass_l and b_twb_mass_u by 
   *  adding the matrix dl and du. Once this is done these values dl and du
   *  are transmitted to the children's b_twb_owed_l and b_twb_owed_u nodes.
  */

  void UpdateBounds(Tree *node, Matrix &dl, Matrix &du){

    // Add dl to mass_l and du to mass_u

    la::AddTo(dl, &node->stat().b_twb_mass_l);
    la::AddTo(du, &node->stat().b_twb_mass_u);

    if(!node->is_leaf()){

      // transmit these values to the children node 

      la::AddTo(dl,&(node->left()->stat().b_twb_owed_l));
      la::AddTo(du,&(node->left()->stat().b_twb_owed_u));

      la::AddTo(dl,&(node->right()->stat().b_twb_owed_l));
      la::AddTo(du,&(node->right()->stat().b_twb_owed_u));
    }
    else{
      
      /* in case of leaf nodes add these values to more_l and more_u */
      la::AddTo(dl,&(node->stat().b_twb_more_l));
      la::AddTo(du,&(node->stat().b_twb_more_u));
    }
  }

  /** This sets the  b_twb_mass_u value at each node of the tree. This value is nothing but the value of B^TWB 
   * with the assumption that the kernel values are all maximum( i.e they are all equal to 1).
   * Alternatively this is equivalent to calculating B^TWB with W being an identity matrix

  */

  void SetMatrixUpperBound(Tree *node){
    
    /*We are claculating the maximum value of B^TWB by assuming kernel value as 1 */
 
    node->stat().b_twb_mass_u.CopyValues(rroot_->stat().b_tb);

    if(!node->is_leaf()){

      /*Not a leaf. Hence recurse */

      SetMatrixUpperBound(node->left());
      SetMatrixUpperBound(node->right());
    }
    else{

      return;
    }
  }

  /** So in this function we update the b_twb_mass_l and b_twb_mass_u values of the parent 
   * by using the values of b_twb_mass_l of the children and the b_twb_mass_u of the children
   * In short one can say something like 
   * lower_of_parent=max (parent, min (children))  
   * upper_of_parent=min(parent, max(children))
   */

  void MergeChildBounds(Regression1Stat *left_stat, Regression1Stat *right_stat, Regression1Stat &parent_stat){
  
    // lower_of_parent=max (parent, min (children))  
    // upper_of_parent=min(parent, max(children))
    
    double sqd_frobenius_norm_left_child=SquaredFrobeniusNorm(left_stat->b_twb_mass_l);
    double sqd_frobenius_norm_right_child=SquaredFrobeniusNorm(right_stat->b_twb_mass_l);
    double sqd_frobenius_norm_parent=SquaredFrobeniusNorm(parent_stat.b_twb_mass_l);
    
    /* will Update the mass_l of the parent */
    if(sqd_frobenius_norm_left_child<sqd_frobenius_norm_right_child){
      
      if(sqd_frobenius_norm_parent<sqd_frobenius_norm_left_child){
	
	parent_stat.b_twb_mass_l.CopyValues(left_stat->b_twb_mass_l);
      }
      else{
	
	/*leave it as such */
      }
    }
    else{ /* right child has a lesser froebius norm */
      if(sqd_frobenius_norm_parent<sqd_frobenius_norm_right_child){
	
	parent_stat.b_twb_mass_l.CopyValues(right_stat->b_twb_mass_l);
      }
      else{
	
	/*leave it as such */
      }
    }
    
    /* will now update mass_u of the parent node */
    sqd_frobenius_norm_left_child=SquaredFrobeniusNorm(left_stat->b_twb_mass_u);
    sqd_frobenius_norm_right_child=SquaredFrobeniusNorm(right_stat->b_twb_mass_u);
    sqd_frobenius_norm_parent=SquaredFrobeniusNorm(parent_stat.b_twb_mass_u);
    
    /* upper= min(parent, max(children)) */
    
    if(sqd_frobenius_norm_left_child > sqd_frobenius_norm_right_child){
      if(sqd_frobenius_norm_parent>sqd_frobenius_norm_left_child){
	
	parent_stat.b_twb_mass_u.CopyValues(left_stat->b_twb_mass_u);
      }
      
      else{
	/* Do nothing */
      }
    }
    else{
      /* Right child has higher frobenius norm */
      if(sqd_frobenius_norm_parent>sqd_frobenius_norm_right_child){
	
	parent_stat.b_twb_mass_u.CopyValues(right_stat->b_twb_mass_u);
      }
      else{
	
	/*leave it as such */
      }
    }
  }

  /** So this function returns a value which is useful for pruning two nodes 
    * It considers the matrix to be a vector and considers it's 1-Norm by summing up 
    * the absolute values of all the elements
    */

  double Compute1NormLike(Matrix &a){
    
    //this function the sum of the absolute values of all the elements in the matrix. 
    //It is like calculating the 1-norm of a vector hence the name of the function 
    
    double value=0.0;
    for(int row=0;row<a.n_rows();row++){

      for(index_t col=0;col<a.n_cols();col++) {
	
	value+=fabs(a.get(row,col));
      }
    }
    return value;
  }

  /** This function checks if the query node and the reference node are 
   *  prunable. 
   */


  int Prunable(Tree *qnode, Tree *rnode, Matrix &dl, Matrix &du, DRange &dsqd_range,DRange &kernel_value_range){

    //This is just the dimensionality of the dataset


    dsqd_range.lo = qnode->bound ().MinDistanceSq (rnode->bound ());
    dsqd_range.hi = qnode->bound ().MaxDistanceSq (rnode->bound ());
    kernel_value_range = kernel_.RangeUnnormOnSq (dsqd_range);

    //The new lower and upper bound after incoporating new info for each dimension 

    /** dl is nothing but the amount of change to the lower bound of the density of the node 
     *supposing pruning occurs. I will calculate dl by multiplying B^TB in the reference node 
     *with the scalar K(QR_max) 
     */

    double min_value=kernel_value_range.lo; 
    double max_value= kernel_value_range.hi;

    //dl <- dl+ min_value(b_tb)=min_value(b_tb). Note the initial value of dl=0

    la:: ScaleOverwrite(min_value, rnode->stat().b_tb, &dl);

    //    la::AddExpert (min_value, rnode->stat().b_tb, &dl); 
    
    /** error in matrix format 
      * The error matrix is constant times B^TB. the constant is defined as below
      */

    double constant= 0.5 * (max_value - min_value);
    Matrix error_matrix;
    la::ScaleInit(constant,rnode->stat().b_tb,&error_matrix);
    double error=SquaredFrobeniusNorm(error_matrix);

    /*du = -B^TB+B^TB*K(QR_min) */

    constant=max_value-1;

    // la::AddExpert (constant, rnode->stat().b_tb, &du); //Note initial value of du=0

    la:: ScaleOverwrite (constant, rnode->stat().b_tb, &du);
    
    /*allowed error is the frobenius norm of the matrix (b_twb_mass_l+dl) */

    Matrix temp;
    la::AddInit (qnode->stat().b_twb_mass_l, dl, &temp); 

    double temp_var=(double)Compute1NormLike(rnode->stat().b_tb)/(double)Compute1NormLike(rroot_->stat().b_tb);
    double allowed_error=tau_* SquaredFrobeniusNorm(temp) * temp_var;

    if(error>=allowed_error){
      /* cannot prune */
      no_prunes++;
      dl.SetAll(0);
      du.SetAll(0);       
      return 0;
    }

    else{
      printf("error is %f\n",error);
      printf("allowed error is %f\n",allowed_error);
      prunes++;
      return 1;
    }
  }

  
  void Regression1Base(Tree *qnode, Tree *rnode){
    
    /* Subtract the B^TB matrix of rnode from B^TWB_more_u vector vector  */
    
    /*qnode->stat().more_u-=rnode->stat().b_tb*/
    
    la::SubFrom(rnode->stat().b_tb, &(qnode->stat().b_twb_more_u));
    
    /* Compute B^TWB exhaustively */
    
    /* Temporary variables */
    
    for(index_t q=qnode->begin();q<qnode->end();q++){
      
      /*For each reference point in rnode */
      const double *q_col = qset_.GetColumnPtr (q);
      
      for(index_t col=rnode->begin();col<rnode->end();col++){
	
	/*form a B^T matrix for each reference point */
	Matrix B_t;
	B_t.Init(rset_.n_rows()+1,1);
	
	/*Form B^T matrix by using one reference point at a time*/
	
	for(index_t row=0;row<rset_.n_rows()+1;row++){
	  
	  if(row==0){
	    
	    B_t.set(row,0,1);
	  }
	  
	  else
	    {
	      
	      B_t.set(row,0,rset_.get(row-1,col));
	    }
	}
	
   	const double *r_col = rset_.GetColumnPtr (col); //get the reference point	
	double dsqd =la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	double val = kernel_.EvalUnnormOnSq (dsqd);
	
	/* temp2 holds B */
	Matrix temp2; 
	la::TransposeInit(B_t,&temp2); 
	
	Matrix temp3;
	la:: MulInit(B_t,temp2,&temp3); //temp3 stores B^TB
	
	
	/* scale temp3 which holds B^TB by val */
	la::Scale(val,&temp3);
	la::AddTo(temp3,&b_twb_l_[q]);
	la::AddTo (temp3,&b_twb_u_[q]);
      }
    }
    
     /** get a tighter lower and upper bound for every dimension by looping over each query point 
     *We now need to go through matrix for each query point and select 
     * the one with the highest and least frobenis squared norm 
     */

    double min_l=DBL_MAX;
    double max_u=DBL_MIN;
    
    // these store the index number of the query point which has 
    //the least and the highest squared frobenius nomrs 

    int max_pointer=qnode->begin();
    int min_pointer=qnode->begin();

    /*Temporary Matrix */
    Matrix temp;
    temp.Init(rset_.n_rows()+1,rset_.n_rows()+1);

    for (index_t q = qnode->begin (); q < qnode->end (); q++){

      //The lower bound on the B^TWB is obtained by 
      //adding b_twb_l[q] to b_twb_more_l. Lets store this sum in temp

      la::AddOverwrite(b_twb_l_[q],qnode->stat().b_twb_more_l,&temp);
      if (SquaredFrobeniusNorm(temp)< min_l){
	
	min_l= SquaredFrobeniusNorm(temp);
	min_pointer=q;
      }

      la::AddOverwrite(b_twb_u_[q],qnode->stat().b_twb_more_u,&temp);
      if (SquaredFrobeniusNorm(temp)> max_u){
	
	max_u=SquaredFrobeniusNorm(temp);
	max_pointer=q;
      }
    }

    // Tighten the lower and upper bounds of B^TWb matrix
    
    /*mass_u=max_frobeniusnorm+more_u */
    /*mass_l=min_frobeniusnorm+more_l */
    
    la::AddOverwrite (b_twb_u_[max_pointer],qnode->stat().b_twb_more_u, &(qnode->stat().b_twb_mass_u));
    la::AddOverwrite (b_twb_l_[min_pointer],qnode->stat().b_twb_more_l, &(qnode->stat().b_twb_mass_l)); 
  }
  
  /** This is the postprocess step. After the dual tree recursion is done
   *  We need to do a Depth first traversal of the tree and at every node 
   *  add the owed value to b_twb_mass values. This is accomplished by calling 
   *  the function UpdateBounds()
   */

  void PostProcess(Tree *qnode){

    UpdateBounds(qnode,qnode->stat().b_twb_owed_l,qnode->stat().b_twb_owed_u);

    if(qnode->is_leaf()){

      Matrix temp1;
      la::AddInit(qnode->stat().b_twb_more_l,qnode->stat().b_twb_more_u, &temp1);

      for(index_t q=qnode->begin();q<qnode->end();q++){
	
	/*b_twb_e=(b_twb_l+b_twb_u_b_twb_more_l+b_twb_more_u)/2 */
       
	Matrix temp2;

	la::AddInit(b_twb_l_[q], b_twb_u_[q], &temp2);
	la::AddOverwrite(temp1,temp2,&(b_twb_e_[q]));

	la::Scale (0.50, &b_twb_e_[q]);
      }
    }
    else{

      /* It is not a leaf node */
      PostProcess(qnode->left());
      PostProcess(qnode->right());    
    }
  }


  /** This is the kind of parent function which will call rest other functions */
  void Regression1(Tree *qnode, Tree *rnode){

    DRange dsqd_range;
    DRange kernel_value_range;

    /* query node statistics */
    Regression1Stat &stat=qnode->stat();

    UpdateBounds(qnode,stat.b_twb_owed_l,stat.b_twb_owed_u);

    /* Since owed_l and owed_u values have been incoprated by calling UpdateBounds() function, set them to 0 */

    stat.b_twb_owed_l.SetAll(0);
    stat.b_twb_owed_u.SetAll(0);


    Regression1Stat *left_stat=NULL;
    Regression1Stat *right_stat=NULL;

    if(!qnode->is_leaf()){

      right_stat= &(qnode->right()->stat());
      left_stat= &(qnode->left()->stat());
    }

    /*try finite difference pruning first  */

    Matrix dl,du;
    dl.Init(rset_.n_rows()+1,rset_.n_rows()+1);
    du.Init(rset_.n_rows()+1,rset_.n_rows()+1);

    /* Initialize it to all 0's */
    dl.SetAll(0);
    du.SetAll(0);

    /** Will check for prunability of the qnode and rnode
      * If they get pruned then we update bounsd and return 
      * Else we will recurse
     */

    if(Prunable(qnode,rnode,dl,du,dsqd_range,kernel_value_range)==1){
      //printf("dl and du received are...\n");
      //dl.PrintDebug();
      //du.PrintDebug();
      UpdateBounds(qnode,dl,du);
      return;
    }
    else
      {

	if(qnode->is_leaf()){

	  if(rnode->is_leaf()){

	    /* This is the Base Case */
	    Regression1Base(qnode,rnode);
	    return;
	  }
	  else{
	    /* rnode is not a leaf node */
	    Tree *rnode_first = NULL, *rnode_second = NULL;
	    BestNodePartners (qnode, rnode->left (), rnode->right (),&rnode_first, &rnode_second);
	    Regression1 (qnode, rnode_first);
	    Regression1 (qnode, rnode_second);
	    return;
	  }
	}

	/* qnode is not a leaf node */
	else{

	  if(rnode->is_leaf()){
	    Tree *qnode_first = NULL, *qnode_second = NULL;
	    BestNodePartners (rnode, qnode->left (), qnode->right (),&qnode_first, &qnode_second);
	    Regression1(qnode_first,rnode);
	    Regression1(qnode_second,rnode);
	  }
	  else{
	    /* Both are non-leaf nodes */
	    Tree *rnode_first = NULL, *rnode_second = NULL;
	    
	    BestNodePartners (qnode->left (), rnode->left (), rnode->right (),&rnode_first, &rnode_second);
	    Regression1 (qnode->left (), rnode_first);
	    Regression1 (qnode->left (), rnode_second);
	    
	    BestNodePartners (qnode->right (), rnode->left (),rnode->right (), &rnode_first, &rnode_second);
	    Regression1 (qnode->right (), rnode_first);
	    Regression1 (qnode->right (), rnode_second);
	  }

	  /* this will now update the bounds of the parent by using the values of the children node*/
	  MergeChildBounds(left_stat,right_stat,stat);
	}
      }
  }

 public:

  /* getter functions */

  Matrix & get_query_dataset(){
    return qset_;
  }

  Matrix& get_reference_dataset(){
    return rset_;
  }

  Matrix & get_results(index_t i){
    return b_twb_e_[i];
  }
  void Compute(){
   
    /** Get the tolerance value from the user */
    tau_=fx_param_double(NULL,"tau",0.5);

    /*Preprocess both the query and reference trees */
    PreProcess(qroot_);
    PreProcess(rroot_);
    /** This sets the value of b_twb_mass_u values in the reference tree */
    SetMatrixUpperBound(qroot_);

    /** Initialize b_twb_u_ */
    for(int q=0;q<qset_.n_cols();q++){

      b_twb_u_[q].CopyValues(rroot_->stat().b_tb);
    }
    /** Start the actual algorithm */
    prunes=no_prunes=0;
    Regression1(qroot_,rroot_);
    PostProcess(qroot_);
    printf("Numbe of prunes are %d\n",prunes);
    printf("Number of no prunes are %d\n",no_prunes);
  } 
  
  void Init(Matrix &query_dataset, Matrix &reference_dataset){
    
    qset_.Alias(query_dataset);
    rset_.Alias(reference_dataset);
    
    /** Scale Dataset if user wants to **/
    
     if (!strcmp (fx_param_str (NULL, "scaling", NULL), "range")){
      
       scale_data_by_minmax (); 
     }
    /* initialize the kernel */
    kernel_.Init (fx_param_double_req (NULL, "bandwidth"));
    
    /* Create trees */  
    qroot_=tree::MakeKdTreeMidpoint < Tree > (qset_, leaflen,&old_from_new_r_,&new_from_old_r_);
    rroot_=tree::MakeKdTreeMidpoint < Tree > (rset_, leaflen,&old_from_new_r_,&new_from_old_r_);
 
    /*Initialize the arraylist b_twb_l_  b_twb_u_ and b_twb_e_ */

    b_twb_l_.Init(qset_.n_cols());
    b_twb_u_.Init(qset_.n_cols());
    b_twb_e_.Init(qset_.n_cols());
    
    int num_of_dimensions=rset_.n_rows();
    for(index_t i=0;i<qset_.n_cols();i++){

      /* Initialize each of them to the size of (D+1)*(D+1) */
      b_twb_l_.Init(num_of_dimensions+1,num_of_dimensions+1);
      b_twb_u_.Init(num_of_dimensions+1,num_of_dimensions+1);
      b_twb_e_.Init(num_of_dimensions+1,num_of_dimensions+1);
   } 

    for(index_t i=0;i<qset_.n_cols();i++){

      /* Initialize each of them to all 0's. However b_twb_u cant be initialized as its value  */
      /*depends on the b_tb value of the root node. hence we initialize this in the compute function */
      b_twb_l_[i].SetAll(0);
      b_twb_e_[i].SetAll(0);

    }
    
    // initialize the kernel
    
    kernel_.Init (fx_param_double_req (NULL, "bandwidth"));
  }

  void Init (){		    
    
    Dataset ref_dataset;

    // read in the number of points owned by a leaf
    //int leaflen = fx_param_int(NULL, "leaflen", 2);

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
       printf("Dataset scaled.....\n");
       scale_data_by_minmax ();
     } 
   
    /* Construct Query and Reference trees */
    rroot_ = tree::MakeKdTreeMidpoint < Tree > (rset_, leaflen, &old_from_new_r_, &new_from_old_r_);
    qroot_=tree::MakeKdTreeMidpoint < Tree > (qset_, leaflen, NULL, NULL);

      
    for(index_t i=0;i<qset_.n_cols();i++){
   
      printf("Old:%d new:%d\n",i,new_from_old_r_[i]);
    }
    /* Initialize b_twb_l_, b_twb_e_ and b_twb_u_ */

    b_twb_l_.Init(qset_.n_cols());
    b_twb_u_.Init(qset_.n_cols());
    b_twb_e_.Init(qset_.n_cols());

    int num_of_dimensions=rset_.n_rows();
    for(index_t i=0;i<qset_.n_cols();i++){
      
      /* Initialize each of them to the size of (D+1)*(D+1) */
      b_twb_l_[i].Init(num_of_dimensions+1,num_of_dimensions+1);
      b_twb_u_[i].Init(num_of_dimensions+1,num_of_dimensions+1);
      b_twb_e_[i].Init(num_of_dimensions+1,num_of_dimensions+1);
    }

    for(index_t i=0;i<qset_.n_cols();i++){

      /* Initialize each of them to all 0's. However b_twb_u cant be initialized as its value  */
      /*depends on the b_tb value of the root node. hence we initialize this in the comute function */
      b_twb_l_[i].SetAll(0);
      b_twb_e_[i].SetAll(0);

    }
    // initialize the kernel

    kernel_.Init (fx_param_double_req (NULL, "bandwidth"));
   
  }

};

#endif




  
