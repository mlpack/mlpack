#include <fastlib/fastlib.h>
#ifndef RANGE_SEARCH_H_
#define RANGE_SEARCH_H_
class RangeSearch{


 public:
  
  // An empty tree stat as we dont store anything.
  /* class TreeStat{
    
  public:
  void Init(const Matrix& matrix, index_t start, index_t count) {
  
  }
  
  
  void Init(const Matrix& matrix, index_t start, index_t count,
  const TreeStat& left, const TreeStat& right) {
  
  Init(matrix, start, count);
  }
  };*/

  /** kd-tree (binary with hrect bounds) with stats for queries. */
  //typedef BinarySpaceTree<DHrectBound<2>, Matrix, TreeStat> QTree;

  /** kd-tree without stats for references. */
  typedef BinarySpaceTree<DHrectBound<2>, Matrix> Tree;
  
  // The query and reference dataset
  
  Matrix qset_;
 
  Matrix rset_;
  
  //Matrix original_qset_;
  //Matrix original_rset_;

  double bandwidth_;
  
  ArrayList<ArrayList <int> > indices_in_range_;

    ArrayList<ArrayList <double> > smoothing_kernel_values_in_range_;

  int num_query_points_;

  int num_ref_points_;

  int num_dims_;

  // TODO: Make this template based

  EpanKernel ek_;


  // The trees

  Tree *qtree_;
  Tree *rtree_;


  int leaf_size_;
  // The permutation matrices as tree building permutes the dataset

  ArrayList<int> new_from_old_q_;

  ArrayList<int> new_from_old_r_;

  ArrayList<int> old_from_new_q_;

  ArrayList<int> old_from_new_r_;

  int num_prunes_;
  

  // Getter functions

  void get_indices_in_range(ArrayList < ArrayList <int> > &result){
    
    // This is a safe operation to do as the range search object
    // doesn't go out of scope
    
    result.InitAlias(indices_in_range_);
  }


  void get_smoothing_kernel_values_in_range(ArrayList < ArrayList <double> > &result){
    result.InitAlias(smoothing_kernel_values_in_range_);
  }

  ~RangeSearch(){

    delete qtree_;
    delete rtree_;

  }

 private:


  void BaseCaseRangeSearch_(Tree *qnode,Tree *rnode){

    //Note the points in qnode and rnode have been permuted.  hence
    //when storing indices of points in range make sure to depermute
    //them.

    int qnode_begin=qnode->begin();
    int qnode_end=qnode->end();

    int rnode_begin=rnode->begin();
    int rnode_end=rnode->end();


    for(int q=qnode_begin;q<qnode_end;q++){// For each query point
      
      double *q_vec=qset_.GetColumnPtr(q);
      
      int original_index_q=old_from_new_q_[q];

      for(int r=rnode_begin;r<rnode_end;r++){
	
	double *r_vec=rset_.GetColumnPtr(r);

	double sqd_dist=la::DistanceSqEuclidean(num_dims_,q_vec,r_vec);
	
	if(sqrt(sqd_dist)<bandwidth_){
	  
	  int original_index_r=old_from_new_r_[r];
	  
	  indices_in_range_[original_index_q].PushBack(1);

	  int size=indices_in_range_[original_index_q].size();
	  indices_in_range_[original_index_q][size-1] =original_index_r;

	  double kernel_value;
	  kernel_value=ek_.EvalUnnormOnSq(sqd_dist);
	  //kernel_value/=ek_.CalcNormConstant(num_dims_);

	  smoothing_kernel_values_in_range_[original_index_q].PushBack(1);
	  smoothing_kernel_values_in_range_[original_index_q][size-1]=
	    kernel_value;
	}
      } 
    } 
  }

  // TODO: may be we can get rid of this function and push it into the
  // base case
  void PerformInclusionPruning_(Tree *qnode,Tree *rnode){

    //Note the points in qnode and rnode have been permuted.  hence
    //when storing indices of points in range make sure to depermute
    //them.

    int qnode_begin=qnode->begin();
    int qnode_end=qnode->end();

    int rnode_begin=rnode->begin();
    int rnode_end=rnode->end();


    for(int q=qnode_begin;q<qnode_end;q++){// For each query point
      
      double *q_vec=qset_.GetColumnPtr(q);
      
      int original_index_q=old_from_new_q_[q];

      for(int r=rnode_begin;r<rnode_end;r++){
	
	double *r_vec=rset_.GetColumnPtr(r);

	double sqd_dist=la::DistanceSqEuclidean(num_dims_,q_vec,r_vec);
		  
	int original_index_r=old_from_new_r_[r];
	
	indices_in_range_[original_index_q].PushBack(1);
	
	int size=indices_in_range_[original_index_q].size();
	indices_in_range_[original_index_q][size-1] =original_index_r;
	
	double kernel_value;
	kernel_value=ek_.EvalUnnormOnSq(sqd_dist);
	//kernel_value/=ek_.CalcNormConstant(num_dims_);
	
	smoothing_kernel_values_in_range_[original_index_q].PushBack(1);
	smoothing_kernel_values_in_range_[original_index_q][size-1]=
	  kernel_value;
      } 
    } 
  }



  void FindCloserNode_(Tree *ref_node, Tree *node1, Tree *node2, 
		       Tree **closer, Tree **farther){
    
    double dist_to_node1=
      ref_node->bound().MinDistanceSq(node1->bound());

    double dist_to_node2=
      ref_node->bound().MinDistanceSq(node2->bound());


    if(dist_to_node1<dist_to_node2){
      
      *closer=node1;
      *farther=node2;
    }
    else{
      
      *closer=node2;
      *farther=node1;
    }
  }

  
  void GNPRangeSearch_(Tree *qnode, Tree *rnode){
    
    // Check if we can prune or not

    double min_dist_sq=qnode->bound().MinDistanceSq(rnode->bound());
    

    if(sqrt(min_dist_sq)>bandwidth_){
      //  This is exclusion pruning
    
      num_prunes_++;
      return;
    }

    double max_dist_sq=qnode->bound().MaxDistanceSq(rnode->bound());

    if(sqrt(max_dist_sq)<bandwidth_){

      // This is inclusion pruning. Simply add all reference
      // contributions.
      
      PerformInclusionPruning_(qnode,rnode);
      return;
    }
    
    Tree *closer,*farther;
  
    // Now we cannot prune hence we recurse.
    
    if(qnode->is_leaf()&&rnode->is_leaf()){
      

      BaseCaseRangeSearch_(qnode,rnode);
      return;
    }

    else{ // atleast one of these nodes is non-leaf

      if(qnode->is_leaf()){
      
	FindCloserNode_(qnode,rnode->left(),
			rnode->right(),&closer,&farther);
	GNPRangeSearch_(qnode,closer);
	GNPRangeSearch_(qnode,farther);
	return;
      }
      else{

	if(rnode->is_leaf()){
	  //Recurse on the left and right child of qnode
	  
	  FindCloserNode_(rnode,qnode->left(),qnode->right(),&closer,&farther);
	  //Recurse on left child of qnode first
	  GNPRangeSearch_(closer,rnode);
	  GNPRangeSearch_(farther,rnode);
	  return;
	}
	else{
	  //Both are non-leaf nodes. hence recurse on left and
	  //children of both trees
	  
	  //Lets begin by recursing using the left child of qnode
	  

	  FindCloserNode_(qnode->left(),rnode->left(),
			  rnode->right(),&closer,&farther);
	  
	  GNPRangeSearch_(qnode->left(),closer);
	  GNPRangeSearch_(qnode->left(),farther);
	  
	  
	// Recurse on the right child of qnode.
	  
	  FindCloserNode_(qnode->right(),rnode->left(),
			  rnode->right(),&closer,&farther);
	  
	  GNPRangeSearch_(qnode->right(),closer);
	  GNPRangeSearch_(qnode->right(),farther);
	  return;
	}
      }
    }
  }
      

  /*  void CompareToNaive_(){ 

      printf("Started naive range search...\n");
      fx_timer_start(NULL,"naive");
      
      ArrayList<ArrayList <int> > indices_in_range; 
      
      ArrayList<ArrayList <double> > smoothing_kernel_values_in_range; 
      
      
      indices_in_range.Init(num_query_points_); 
      
      smoothing_kernel_values_in_range.Init(num_query_points_);
      
      
      for(int i=0;i<num_query_points_;i++){ 
	
	indices_in_range[i].Init(0); 
	smoothing_kernel_values_in_range[i].Init(0); 
      } 
      
      
      for(int q=0;q<num_query_points_;q++){ 
	
	double *q_vec=original_qset_.GetColumnPtr(q);
	for(int r=0;r<num_ref_points_;r++){ 
	  
	  double *r_vec=original_rset_.GetColumnPtr(r); 
	  
 	double sqd_dist=la::DistanceSqEuclidean(num_dims_,q_vec,r_vec); 
	
 	if(sqrt(sqd_dist)<bandwidth_){ 
	  
 	  indices_in_range[q].PushBack(1); 
 	  int size=indices_in_range[q].size(); 
 	  indices_in_range[q][size-1] =r; 
	  
 	  double kernel_value; 
 	  kernel_value=ek_.EvalUnnormOnSq(sqd_dist); 
 	  kernel_value/=ek_.CalcNormConstant(num_dims_); 
	  
 	  smoothing_kernel_values_in_range[q].PushBack(1); 
 	  smoothing_kernel_values_in_range[q][size-1]= 
	     	    kernel_value; 
	  
 	} 
       } 
       } 
      printf("Finished naive range search..\n");
      fx_timer_stop(NULL,"naive");
   
      FILE *hp; 
      hp=fopen("range_search_naive.txt","w"); 
      
      for(int q=0;q<num_query_points_;q++){  
      
      int size=indices_in_range[q].size(); 
      
      
      for(int i=0;i<size;i++){ 
      
      fprintf(hp,"indices_in_range[%d][%d]=%d,smoothing_kernel_value[%d][%d]=%f..\n", 
      q,i,indices_in_range[q][i],q,i, 
      smoothing_kernel_values_in_range[q][i]); 
      
      } 
      }
      }*/


public: 
  
  void PerformRangeSearch(){


    fx_timer_start(NULL,"fast");
    
    GNPRangeSearch_(qtree_,rtree_);
    
    fx_timer_stop(NULL,"fast");

    // Having finished lets check the kernel values and indices
    
    /* FILE *gp;
    gp=fopen("range_search_fast.txt","w");
    for(int q=0;q<num_query_points_;q++){ // Let this index the unpermuted set
      
      int size=indices_in_range_[q].size();
      
      
      for(int i=0;i<size;i++){
	
	fprintf(gp,"indices_in_range[%d][%d]=%d,smoothing_kernel_value[%d][%d]=%f..\n",
		q,i,indices_in_range_[q][i],q,i,
		smoothing_kernel_values_in_range_[q][i]);
	
      }
      }*/
    
    //    printf("Number of prunes are =%d..\n",num_prunes_);
    
    //printf("Finished writing to file for fast range search..\n");
    //CompareToNaive_();
   
  }
  
  void Init(Matrix &ref_data,Matrix &query_data,
	    double smoothing_kernel_bandwidth){
    
    //Copy all the data. Do not alias data here because the data will
    //get permuted during tree formation

    
    rset_.Copy(ref_data);
    
    qset_.Copy(query_data);
    
    // Copy the data, This is only for naive calculations

    //original_qset_.Alias(query_data);
    //original_rset_.Alias(ref_data);
    
    num_query_points_=qset_.n_cols();
    
    num_ref_points_=rset_.n_cols();
    
    bandwidth_=smoothing_kernel_bandwidth; 
    
    num_dims_=rset_.n_rows();
    
    if(min(num_query_points_,num_ref_points_)>1000){
      leaf_size_=max(10.0,ceil(50*log(min(num_query_points_,num_ref_points_))));
    }
    else{
      
      //leaf_size_=max(10.0,ceil(50*log(min(num_query_points_,num_ref_points_))));
      leaf_size_=20;
    }

  // Lets build the trees
    
    fx_timer_start(NULL,"tree_build");
    
    qtree_ = 
      tree::MakeKdTreeMidpoint<Tree>(qset_, leaf_size_, 
				     &old_from_new_q_, &new_from_old_q_);
    
    rtree_ = 
      tree::MakeKdTreeMidpoint<Tree>(rset_, leaf_size_, 
				     &old_from_new_r_, &new_from_old_r_);
    fx_timer_stop(NULL,"tree_build");
    
    // We can also initialize the arraylists
    
    indices_in_range_.Init(num_query_points_);
    
    smoothing_kernel_values_in_range_.Init(num_query_points_);
    
    //Initialize these arraylists
    
    for(int i=0;i<num_query_points_;i++){
      
      indices_in_range_[i].Init(0);
      smoothing_kernel_values_in_range_[i].Init(0);
    }
    
    num_prunes_=0;
    
    // Initialize the epan kernel with the bandwidth provided
    
    ek_.Init(bandwidth_);
  }  
};
#endif
