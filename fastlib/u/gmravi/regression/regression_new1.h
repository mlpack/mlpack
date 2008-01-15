/* THIS FILE COMPUTES B^TWB by pruning the entire matrix */


#ifndef REGRESSION_NEW1_H
#define REGRESSION_NEW1_H
#define leaflen 2
#define MAXDOUBLE 32768.0


  /* This is a friend function and will be used by both the classes defined in this file*/

 double SquaredFrobeniusNorm(Matrix &a){
   
   double sqd_frobenius_norm=0;
   for(int rows=0;rows<a.n_rows();rows++){
     
     for(int cols=0;cols<a.n_cols();cols++){
       sqd_frobenius_norm+=pow(a.get(rows,cols),2);
     }
   }
   return sqd_frobenius_norm;
 }


template <typename TKernel> class NaiveRegression_new1{

 private:

  /** query dataset */
  Matrix qset_;

  /** reference dataset */
  Matrix rset_;

  /** kernel */
  TKernel kernel_;

  /**results stores the Matrix of each query point**/
  ArrayList <Matrix> results_;

 public:

  /*Interesting functions................. */

  /* This is a friend function adn will be used by both the naive class and the fast class */

  double friend SquaredFrobeniusNorm(Matrix &a);

  void CompareWithFast(ArrayList<Matrix> &fast){

    /*Compare the relative frobenius norm */
    double sqdfast,sqdnaive,relative_error,max_error=0.0;
    for(index_t i=0;i<qset_.n_cols();i++){
       sqdfast=SquaredFrobeniusNorm(fast[i]);
       sqdnaive=SquaredFrobeniusNorm(results_[i]);
       printf("sqdfast=%f and sqdnaive=%f\n",sqdfast,sqdnaive);
       relative_error=fabs(sqdfast-sqdnaive)/sqdnaive;
       if(max_error<=relative_error){

	 max_error=relative_error;
       }
    }
    printf("Maximum error was %f\n",max_error);
    printf("The B^TWB matrices are..\n");
    for(index_t q=0;q<qset_.n_cols();q++){

      printf("Fast..q=%d\n",q);
      fast[q].PrintDebug();
      printf("Naive..q=%d\n",q);
      results_[q].PrintDebug();
    }

  }

  void Compute (){
    
    Matrix B_t;
    B_t.Init(rset_.n_rows()+1,rset_.n_cols());
    
    for(index_t row=0;row<rset_.n_rows()+1;row++){
      
      for(index_t col=0;col<rset_.n_cols();col++){
	
	if(row==0){
	  B_t.set(row,col,1);
	}
	else
	  {
	    B_t.set(row,col,rset_.get(row-1,col));
	  }
      }
    }

    /*temp2 hods B */
    Matrix temp2; 
    la::TransposeInit(B_t,&temp2); 
    
    for(index_t q=0;q<qset_.n_cols();q++){
      
      const double *q_col = qset_.GetColumnPtr (q); //get the query point
     
      /*Form a diagonal W vector */

      Matrix W;
      Vector v;
      v.Init(rset_.n_cols());
     
      for(index_t i=0;i<rset_.n_cols();i++){
	
	const double *r_col = rset_.GetColumnPtr (i); //get the reference point
	double dsqd =la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	double val = kernel_.EvalUnnormOnSq (dsqd);
	v[i]=val;
      }

      W.InitDiagonal(v);
      Matrix temp3;
      la:: MulInit(B_t,W,&temp3); //temp3 stores B^TW
      la::MulOverwrite(temp3,temp2,&results_[q]);
  
    }
  }
    
  void Init (Matrix query_dataset, Matrix reference_dataset){

   
    qset_.Copy(query_dataset);
    rset_.Copy(reference_dataset);
    
    kernel_.Init (fx_param_double_req (NULL, "bandwidth"));

    //Initialize results_
    results_.Init (qset_.n_cols ());
    
    for (index_t i = 0; i < qset_.n_cols (); i++){
      
      results_[i].Init(rset_.n_rows()+1,rset_.n_rows()+1); //A square matrix
    }
    
     
  }
};  //Naive regression1 is done

template <typename TKernel> class Regression_new1{

 public:

  //forward declaration of Regression_new1 class

  class Regression1Stat;

  //Our binary tree uses Regression1Stat

  typedef BinarySpaceTree < DHrectBound <2>, Matrix, Regression1Stat > Tree;

  class Regression1Stat{

  public: 

    /*The B^TB matrix. This is required to be stored in the reference tree*/

    Matrix b_tb;

    /** lower bound on the B^TWB matrix for the query points owned by this node */    
    Matrix b_twb_mass_l;
    
    /**
     * additional offset for the lower bound on the B^TWB for the query
     * points owned by this node (for leaf nodes only).*/
    
    Matrix b_twb_more_l;
    
    /* lower bound offset Matrix passed from above*/
    
    Matrix b_twb_owed_l;
    
    /** upper bound on the B^TWB matrix for the query points owned by this node. 
     */
    
    Matrix b_twb_mass_u;
    
    /**
     * additional offset for the upper bound on the densities for the query
     * points owned by this node (for leaf nodes only). 
     */
    Matrix b_twb_more_u;

    /** upper bound offset passed from above. 
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
  
  /* B^TWB matrix for each query point*/
  ArrayList <Matrix> b_twb_e_;
  
  /* upper bound on the B^TWB matrix*/
  ArrayList <Matrix> b_twb_u_;
   
  /** accuracy parameter */
  double tau_;

  /** Regression values for the reference points */
  Vector rset_weights_;

  /** Mappings from old dataset to new dataset */
  ArrayList <int> old_from_new_r_;
  ArrayList <int> new_from_old_r_;


  /* Interesting functions */

  /*This is a friend function which will be used by bothe the classes defined in the file */

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


  void PreProcess(Tree *node){
    
    int num_of_dimensions=rset_.n_rows();

    /*First initalize the matrices*/

    node->stat().b_tb.Init(num_of_dimensions+1,num_of_dimensions+1);
    node->stat().b_twb_mass_l.Init(num_of_dimensions+1,num_of_dimensions+1);
    node->stat().b_twb_mass_u.Init(num_of_dimensions+1,num_of_dimensions+1);

    node->stat().b_twb_more_l.Init(num_of_dimensions+1,num_of_dimensions+1);
    node->stat().b_twb_more_u.Init(num_of_dimensions+1,num_of_dimensions+1);

    node->stat().b_twb_owed_l.Init(num_of_dimensions+1,num_of_dimensions+1);
    node->stat().b_twb_owed_u.Init(num_of_dimensions+1,num_of_dimensions+1);


    /**all lower bound matices will be all 0's **/

    node->stat().b_twb_mass_l.SetAll(0);
    node->stat().b_twb_more_l.SetAll(0); 
    node->stat().b_twb_owed_l.SetAll(0);


    /**Set the upper bound matrix owed_u and more_u to 0. The upper bound matrix mass_u will be set later*/

    node->stat().b_twb_more_u.SetAll(0); 
    node->stat().b_twb_owed_u.SetAll(0);


    /**Base  Case*/

    if(node->is_leaf()){
    
      node->stat().b_twb_more_l.SetAll(0);
      node->stat().b_twb_more_u.SetAll(0);

      //Fill the B^tB matrix. Note B^TB is a symmetric matrix. Hence we will fill up just the upper triangular matix and then invert along the diagonal
    


      /*First from the B matrix */
      /* Use temp as a temporary matrix variable. Which has the points of rnode in a matrix format */

      Matrix temp;
      index_t span=node->end()-node->begin();
      rset_.MakeColumnSlice(node->begin(),span,&temp);
      Matrix B_t;
      B_t.Init(rset_.n_rows()+1,node->end()-node->begin());

      for(index_t row=0;row<rset_.n_rows()+1;row++){

	for(index_t col=0;col<node->end()-node->begin();col++){

	  if(row==0){
	    B_t.set(row,col,1);
	  }
	  else
	    {
	      B_t.set(row,col,temp.get(row-1,col));
	    }
	}
      }
      /*printf("Matrix B_t is \n");
	B_t.PrintDebug(); */

      Matrix temp2; //temp2 hold B^T
      la::TransposeInit(B_t,&temp2);
      la:: MulOverwrite(B_t,temp2,&(node->stat().b_tb));
    }
     
      else{
	/* for non-leaf recurse */
	
	PreProcess(node->left());
	PreProcess(node->right());
	la::AddOverwrite (node->left()->stat().b_tb, node->right()->stat().b_tb, &(node->stat().b_tb));

	
      }
    }

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

  void UpdateBounds(Tree *node, Matrix &dl, Matrix &du){

    /** Add dl to mass_l and du to mass_u*/
    la::AddTo(dl, &node->stat().b_twb_mass_l);
    la::AddTo(du, &node->stat().b_twb_mass_u);

    if(!node->is_leaf()){

      /** transmit these values to the children node */
      la::AddTo(dl,&(node->left()->stat().b_twb_owed_l));
      la::AddTo(du,&(node->right()->stat().b_twb_owed_u));
    }
    else{
      
      /* in case of leaf nodes add these values to more_l and more_u */
      la::AddTo(dl,&(node->stat().b_twb_more_l));
      la::AddTo(du,&(node->stat().b_twb_more_u));
    }
  }

  void SetMatrixUpperBound(Tree *node){
    
    /*We are claculating the maximum value of B^TWB by assuming kernel value as 1 */
 
    node->stat().b_twb_mass_u.CopyValues(rroot_->stat().b_tb);

    if(!node->is_leaf()){

      SetMatrixUpperBound(node->left());
      SetMatrixUpperBound(node->right());
    }
    else{

      return;
    }
  }



  void MergeChildBounds(Regression1Stat *left_stat, Regression1Stat *right_stat, Regression1Stat &parent_stat){
  

      /* CHECK THIS................... */
      /* lower=max (parent, min (children)  upper=min(parent, max(children))*/
    
      
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

  double Compute1NormLike(Matrix &a){
    
    /* this function the sum of the absolute values of all the elements in the matrix. It is like calculating the 1-norm of a vector hence the name of the function */
    
    double value=0;
    for(int row=0;row<a.n_rows();row++){

      for(index_t col=0;col<a.n_cols();col++) {
	
	value+=fabs(a.get(row,col));
      }
    }
    return value;
  }

  int Prunable(Tree *qnode, Tree *rnode, Matrix &dl, Matrix &du, DRange &dsqd_range,DRange &kernel_value_range){

    int num_of_dimensions=rset_.n_rows();

    dsqd_range.lo = qnode->bound ().MinDistanceSq (rnode->bound ());
    dsqd_range.hi = qnode->bound ().MaxDistanceSq (rnode->bound ());
    kernel_value_range = kernel_.RangeUnnormOnSq (dsqd_range);

    /* The new lower and upper bound after incoporating new info for each dimension CHECK THIS......*/

    /* I will calculate dl by multiplying B^TB in the reference node with the scalar K(QR_max) */

    double min_value=kernel_value_range.lo; 
    double max_value= kernel_value_range.hi;

    la::AddExpert (min_value, rnode->stat().b_tb, &dl);  //dl <- dl+ min_value(b_tb)=min_value(b_tb)
    
    /*  error in matrix format */

    Matrix error_matrix;
    error_matrix.Init(num_of_dimensions+1,num_of_dimensions+1);
    error_matrix.SetAll(0);
    
    double constant= 0.5 * (max_value - min_value);

    /* The error matrix is constant times B^TB.   CHECK THIS */

    /* Hence squared frobenius norm of error matrix is constant^2 * sqdfrobeniusnorm(error_matrix) */


    double error=pow(constant,2)*SquaredFrobeniusNorm(rnode->stat().b_tb);

    /*du = -B^TB+B^TB*K(QR_min) */

    constant=max_value-1;

    la::AddExpert (constant, rnode->stat().b_tb, &du);

    /*allowed error is the frobenius norm of the matrix (b_twb_mass_l+dl) */

    Matrix temp;
    la::AddInit (qnode->stat().b_twb_mass_l, dl, &temp); 

    double temp_var=(double)Compute1NormLike(rnode->stat().b_tb)/Compute1NormLike(rroot_->stat().b_tb);
    double allowed_error=tau_* SquaredFrobeniusNorm(temp) * temp_var;
   
    if(error>allowed_error){
      /* cannot prune */

      printf("Sorry cannot prune for this node %d\n",qnode->begin());
      printf("error is %f\n",error);
      printf("allowed error is %f\n",allowed_error);

      dl.SetAll(0);
      du.SetAll(0);       
      return 0;
    }

    else{
      printf("Hence can prune for this node %d\n",qnode->begin());
      printf("error is %f\n",error);
      printf("allowed error is %f\n",allowed_error);
      return 1;

    }
  }

  
  void Regression1Base(Tree *qnode, Tree *rnode){


    printf("Hit regression base..\n");
    /* Subtract the B^TB matrix of rnode from B^TWB_more_u vector vector  */
    
    /*qnode->stat().more_u-=rnode->stat().b_tb*/

    la::SubFrom(rnode->stat().b_tb, &(qnode->stat().b_twb_more_u));

    /* Compute B^TWB exhaustively */

    /* Temporary variables */

    for(index_t q=qnode->begin();q<qnode->end();q++){

      const double *q_col = qset_.GetColumnPtr (q); //get the query point
      Matrix temp;
      index_t span=rnode->end()-rnode->begin();
      rset_.MakeColumnSlice(rnode->begin(),span,&temp);
      Matrix B_t;
      B_t.Init(rset_.n_rows()+1,rnode->end()-rnode->begin());

      for(index_t row=0;row<rset_.n_rows()+1;row++){

	for(index_t col=0;col<rnode->end()-rnode->begin();col++){

	  if(row==0){
	    B_t.set(row,col,1);
	  }
	  else
	    {
	      B_t.set(row,col,temp.get(row-1,col));
	    }
	}
      }

      /* temp2 holds B */
      Matrix temp2; 
      la::TransposeInit(B_t,&temp2); 
   
      Vector v;
      v.Init(rnode->end()-rnode->begin());
     
      index_t t=0;
      for(index_t i=rnode->begin();i<rnode->end();i++){

	const double *r_col = rset_.GetColumnPtr (i); //get the reference point
	double dsqd =la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	double val = kernel_.EvalUnnormOnSq (dsqd);
	v[t++]=val;

      }

      printf("V vector formed..\n");
      /*Form a diagonal W vector */

      Matrix W;
      W.InitDiagonal(v);
    
      Matrix temp3;
      // temp3.Init(rset_.n_rows()+1,rnode->end()-rnode->begin());
      //printf("temp3 set up..\n");
    
      la:: MulInit(B_t,W,&temp3); //temp3 stores B^TW
      printf("1 done..\n");

      /* temp4 holds b_twb which should be added to the exisitng value of b_twb */
      Matrix temp4;
      la::MulInit(temp3,temp2,&temp4);
    

      printf("Initially B^TWB lower is ..\n");
      b_twb_l_[q].PrintDebug();    

      printf("Initially B^TWB upper is ..\n");
      b_twb_u_[q].PrintDebug();

      la::AddTo (temp4,&b_twb_l_[q]); //temp4 holds B^TWB
      la::AddTo (temp4,&b_twb_u_[q]);

      printf("B_TWB_lower for node %d\n",q);
      b_twb_l_[q].PrintDebug();

      printf("finally B_TWB upper for node %d \n",q);
      b_twb_u_[q].PrintDebug();
    }
    
    /* get a tighter lower and upper bound for every dimension by looping over each query point */
    /*We now need to go through matrix for each query point and select the one with the highest and least frobenis squared norm */

    double min_l=MAXDOUBLE;
    double max_u=-1*MAXDOUBLE;

    /* these store the index number of the query point which has the least and the highest squared frobenius nomrs */

    int max_pointer=qnode->begin();
    int min_pointer=qnode->begin();

    /*Temporary Matrix */
    Matrix temp;
    temp.Init(rset_.n_rows()+1,rset_.n_rows()+1);

    for (index_t q = qnode->begin (); q < qnode->end (); q++){

      la::AddOverwrite(b_twb_l_[q],qnode->stat().b_twb_more_l,&temp);
      if (SquaredFrobeniusNorm(temp)< min_l){
	
	min_l= SquaredFrobeniusNorm(temp);
	min_pointer=q;
      }

      la::AddOverwrite(b_twb_u_[q],qnode->stat().b_twb_more_u,&temp);
      if (SquaredFrobeniusNorm(temp)> max_u){
	
	max_u =SquaredFrobeniusNorm(temp);
	max_pointer=q;
      }
    }

      // Tighten the lower and upper bounds of B^TWb matrix

      /*mass_u=max_frobeniusnorm+more_u */
      /*mass_l=min_frobeniusnorm+more_l */

      la::AddOverwrite (b_twb_u_[max_pointer],qnode->stat().b_twb_more_u, &(qnode->stat().b_twb_mass_u));
      la::AddOverwrite (b_twb_l_[min_pointer],qnode->stat().b_twb_more_l, &(qnode->stat().b_twb_mass_l)); 
  }

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
    dl.Init(rset_.n_rows()+1,qset_.n_rows()+1);
    du.Init(rset_.n_rows()+1,qset_.n_rows()+1);

    /* Initialize it to all 0's */
    dl.SetAll(0);
    du.SetAll(0);

    if(Prunable(qnode,rnode,dl,du,dsqd_range,kernel_value_range)==1){
      
      UpdateBounds(qnode,dl,du);
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
    tau_=fx_param_double(NULL,"tau",0.2);

    /** Preprocess both the query and reference trees */
    PreProcess(qroot_);
    PreProcess(rroot_);
    /** This sets the value of b_twb_mass_u values in the reference tree */
    SetMatrixUpperBound(qroot_);

    /** Initialize b_twb_u_ */
    for(int q=0;q<qset_.n_cols();q++){

      b_twb_u_[q].CopyValues(rroot_->stat().b_tb);
    }
    /** Start the actual algorithm */
    Regression1(qroot_,rroot_);

    PostProcess(qroot_);
    
  } 
  
  void Init(Matrix &query_dataset, Matrix &reference_dataset){
    
    qset_.Alias(query_dataset);
    rset_.Alias(reference_dataset);
    
    /** Scale Dataset if user wants to **/
    
    /*  if (!strcmp (fx_param_str (NULL, "scaling", NULL), "range")){
      
      scale_data_by_minmax (); 
      }*/ 
    /* initialize the kernel */
    kernel_.Init (fx_param_double_req (NULL, "bandwidth"));
    
    /* Create trees */  
    qroot_=tree::MakeKdTreeMidpoint < Tree > (qset_, leaflen,&old_from_new_r_,&new_from_old_r_);
    rroot_=tree::MakeKdTreeMidpoint < Tree > (rset_, leaflen,&old_from_new_r_,&new_from_old_r_);

    printf("new_from_old_r_ is ...\n");

   
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
      /*depends on the b_tb value of the root node. hence we initialize this in the comute function */
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

       scale_data_by_minmax ();
     } 
   
    /* Construct Query and Reference trees */
    rroot_ = tree::MakeKdTreeMidpoint < Tree > (rset_, leaflen, &old_from_new_r_, &new_from_old_r_);
    qroot_=tree::MakeKdTreeMidpoint < Tree > (qset_, leaflen, NULL, NULL);

    printf("The trees are as follows \n");
    qroot_->Print();

    
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




  
