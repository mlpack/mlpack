/**

 */

#ifndef THOR_2PT_H
#define THOR_2PT_H

#include <fastlib/fastlib.h>
#include "fastlib/thor/thor.h"
#include "two_point.h"
#include "three_point.h"
#include "dfs3.h"
#include "metric.h"

/**
 * THOR-based N-Point Correlation Functions
 */

const fx_entry_doc module_entries[] = {  
  {"bins", FX_PARAM, FX_STR, NULL,
   "Name of file which specifies range of bins for two point correlation. Each bin is assumed to begin where previous bin ends, e.g. '1 2 5' will give counts in ranges 1-2 and 2-5."}, 
  {"red", FX_PARAM, FX_INT, NULL,
   "red=1 means program will use redshift in computing correlation. red=0 or unspecified means redshift will not be used. Option is ignored if using Cartesian coordinates."},
  {"cart", FX_PARAM, FX_INT, NULL,
   "cart=1 specifies cartesian coordinates, of arbitrary dimensionality. cart=0 (or unused) will use projected spherical coordinates, assumed to be in radians."},
  {"naive", FX_PARAM, FX_INT, NULL,
   "naive=1 performs naive computation, for the purpose of testing results of dual-tree algorithm. naive=0 (or unused) will use dual-tree method. No approximations are made in tree method, so both should return the same results."},
  FX_ENTRY_DOC_DONE  
};

const fx_module_doc param_doc = {
  module_entries, NULL, 
  "Parameters of Simulated System \n"
};


class Thor3PC {
  
  static const int POS = 0;
  static const int RED = 1;
  static const int LUM = 2;
    
 public:
  
  /** the bounding type which is required by THOR */
  typedef DHrectBound<2> Bound;
  
  class Thor3PCStat;
  class Thor3PCPoint;
  
  typedef Thor3PCPoint QPoint;
  typedef Thor3PCPoint RPoint;
  
  
  /** query stat */
  typedef Thor3PCStat QStat;
  
  /** reference stat */
  typedef Thor3PCStat RStat;

  /** query node */
  typedef ThorNode<Bound, QStat> QNode;

  /** reference node */
  typedef ThorNode<Bound, RStat> RNode;


  /** parameter class */
  class Param {
  public:
   
    index_t query_count_;
    index_t reference_count_; 
    int redshift_;
    int cartesian_;
    int auto_corr_;
    Vector bounds_;
   
    OT_DEF(Param) {          
      OT_MY_OBJECT(query_count_);
      OT_MY_OBJECT(reference_count_);
      OT_MY_OBJECT(bounds_);
      OT_MY_OBJECT(redshift_);
    }
  public:
    
    /**   
     * Initializes parameters from a data node (Req THOR).
     */
    void Init(datanode *module) {
      auto_corr_ = 1;
      Matrix temp_bounds;
      const char* fp_bounds;
      fp_bounds = fx_param_str_req(module, "bins");
      data::Load(fp_bounds, &temp_bounds);   
      bounds_.Init(temp_bounds.n_rows());
      for (int i = 0; i < temp_bounds.n_rows(); i++){
	bounds_[i] = temp_bounds.get(i,0)*temp_bounds.get(i,0);
      }   
      redshift_ = fx_param_int(module, "red", 0);
      cartesian_ = fx_param_int(module, "cart", 0);
    }
    
    void FinalizeInit(datanode *module, int dimension) {
         
    }

    void GetBins(Vector& bins_out){
      bins_out.Init(bounds_.length());
      bins_out.CopyValues(bounds_);
    }

    void SetAuto(){
      auto_corr_ = 0;
    }

    int Auto() const{
      return auto_corr_;
    }
  };
  
  /** 
   * the type of each KDE point - this assumes that each query and
   * each reference point is appended with a weight.
   */
  class Thor3PCPoint {
  public:
    
    /** the point's position */
    Vector pos_;
    index_t old_index_;
    OT_DEF(Thor3PCPoint) {    
      OT_MY_OBJECT(pos_);
      OT_MY_OBJECT(old_index_);
    }
    
  public:
    
    /** getters for the vector so that the tree-builder can access it */   
    const Vector& vec() const { return pos_; }

    Vector& vec() { return pos_; }
    
    /** initializes all memory for a point */
    void Init(const Param& param, const DatasetInfo& schema) {
      if (param.cartesian_){
	pos_.Init(schema.n_features());
      } else{
	if(param.redshift_){
	  pos_.Init(3);     
	} else {
	  pos_.Init(2);
	}
      }
    }
    
    /** 
     * sets contents assuming all space has been allocated.
     * Any attempt to allocate memory here will lead to a core dump.
     */
    void Set(const Param& param, index_t index, Vector& data) {          
      for (int i = 0; i < pos_.length(); i++){
	pos_[i] = data[i];
      }     
      old_index_ = index;
    }      

};


  
  /**
   * Per-node bottom-up statistic for both queries and references.
   *
   * The statistic must be commutative and associative, thus bottom-up
   * computable.
   *
   */
  class Thor3PCStat {
    
  public:    
   
    OT_DEF(Thor3PCStat) {    
    }
    
    /**
     * Initialize to a default zero value, as if no data is seen (Req THOR).
     *
     * This is the only method in which memory allocation can occur.
     */
  public:
    void Init(const Param& param) {     
    }
    
    /**
     * Accumulate data from a single point (Req THOR).
     */
    void Accumulate(const Param& param, const Thor3PCPoint& point) {    
    }
    
    /**
     * Accumulate data from one of your children (Req THOR).
     */
    void Accumulate(const Param& param, const Thor3PCStat& child_stat, 
		    const Bound& bound, index_t child_n_points) {     
    }
    
    /**
     * Finish accumulating data; for instance, for mean, divide by the
     * number of points.
     */
    void Postprocess(const Param& param, const Bound& bound, index_t n) {     
    }
  };
  


  /**
   * Coarse result on a region.
   * Presently unused?
   */
  class Delta {
  public:

    /** Density update to apply to children's bound */
  
    OT_DEF_BASIC(Delta) {
    }

  public:
    void Init(const Param& param) {
    }
  };

  /** coarse result on a region */
  // This is unused for CNA, as all we can do is prune or recurse.
  class QPostponed {
  public:

  
    OT_DEF_BASIC(QPostponed) {    
    }

  public:
    
    /** initialize postponed information to zero */
    void Init(const Param& param) {     
    }

    void Reset(const Param& param) {
     
    }

    /** accumulate postponed information passed down from above */
    void ApplyPostponed(const Param& param, const QPostponed& other) {     
    }
  };

  /** individual query result */
  class QResult {
  public:   

  
    OT_DEF_BASIC(QResult) {   
    }

  public:
    void Init(const Param& param) {              
    }

    void Seed(const Param& param, const QPoint& q) {
    }

   

    // Apply accelration to query point, and reset velocity.
    void Postprocess(const Param& param, const QPoint& q, index_t q_index,
		     const RNode& r_root) {    
    }   
  

    /** apply left over postponed contributions */
    void ApplyPostponed(const Param& param, const QPostponed& postponed,
			const QPoint& q, index_t q_index) {
     
    }
  };

  class QSummaryResult {
  public:

  
    OT_DEF_BASIC(QSummaryResult) {
     
    }

  public:
    
    /** initialize summary result to zeros */
    void Init(const Param& param) {
     
    }

    void Seed(const Param& param, const QNode& q_node) {

    }

    void StartReaccumulate(const Param& param, const QNode& q_node) {
    
    }

    /** 
     * refine query summary results by incorporating the given current
     * query result
     */
    void Accumulate(const Param& param, const QResult& result) {
    
    }

    /** 
     * this is the vertical operator that refines the current query summary
     * results based on the summary results owned by the given child
     */
    void Accumulate(const Param& param,
		    const QSummaryResult& result, index_t n_points) {
     
    }

    void FinishReaccumulate(const Param& param, const QNode& q_node) {
    }

    /** 
     * horizontal join operator that accumulates the current best guess
     * on the density bound on the reference portion that has not been
     * visited so far.
     */
    void ApplySummaryResult(const Param& param,
			    const QSummaryResult& summary_result) {
     
    }

    /** apply deltas */
    void ApplyDelta(const Param& param, const Delta& delta) {
     
    }

    /** apply postponed contributions that were passed down */
    void ApplyPostponed(const Param& param,
			const QPostponed& postponed, const QNode& q_node) {    
    }

  };

  /**
   * A simple postprocess-step global result.
   * This will store thermodynamic quantities for the whole system.
   */
  class GlobalResult {
  public:

    TwoPoint two_point_;
    ThreePoint three_point_;

    OT_DEF_BASIC(GlobalResult) {        
      OT_MY_OBJECT(two_point_);
      OT_MY_OBJECT(three_point_);
    }

  public:
    void Init(const Param& param) { 
      two_point_.Init(param.bounds_);
      three_point_.Init(param.bounds_);
    }

    void Accumulate(const Param& param, const GlobalResult& other) {      
      two_point_.Merge(other.two_point_);
      three_point_.Merge(other.three_point_);
    }

    void ApplyDelta(const Param& param, const Delta& delta) {}
    void UndoDelta(const Param& param, const Delta& delta) {}

    void Postprocess(const Param& param) {     
    }

    void Report(const Param& param, datanode *datanode) {
    }
    void ApplyResult(const Param& param, const QPoint& q_point, index_t q_i,
		     const QResult& result) {    
    }
  };
  
  /**
   * Abstract out the inner loop in a way that allows temporary variables
   * to be register-allocated.
   */
  class PairVisitor {
  public:       
    TwoPoint local_two_;   
    ThreePoint local_three_;
  private:
    
    
  public:
    void Init(const Param& param) {       
      local_two_.Init(param.bounds_);
      local_three_.Init(param.bounds_);
    }
    
    /** apply single-tree based pruning by iterating over each query point
     */
    bool StartVisitingQueryPoint
      (const Param& param, const QPoint& q, index_t q_index,
       const RNode& r_node, const Delta& delta,
       const QSummaryResult& unapplied_summary_results, QResult* q_result,
       GlobalResult* global_result) {         
      
      double bound;
      if (param.cartesian_){
	bound = r_node.bound().MinDistanceSq(q.pos_);
      } else{
	if(param.redshift_){
	  bound = mtrc::MinRedShiftDistSq(r_node.bound(), q.pos_,
					  param.redshift_val_);
	} else {
	  bound = mtrc::MinSphereDistSq(r_node.bound(), q.pos_);      
	}
      }
      if (bound > local_two_.Max()) {
	return false;
      }    
      
      // otherwise, we need to iterate over each reference point     
      return true;
    }


    /** exhaustive computation between a query point and a reference point
     */
    void VisitPair(const Param& param, const QPoint& q, index_t q_index,
		   const RPoint& r, index_t r_index) {   
      if (unlikely((q_index == r_index) && param.Auto() )){
	return;
      }
      double dist;
      if (param.cartesian_){
	dist = la::DistanceSqEuclidean(q.pos_, r.pos_);
      } else {
	if (param.redshift_){
	  dist = mtrc::RedShiftDistSq(q.pos_, r.pos_, param.redshift_val_);
	} else {
	  dist = mtrc::SphereDistSq(q.pos_, r.pos_);
	}
      }
      local_two_.Add(dist);      
    }


    void VisitTriple(const Param& param, const QPoint& q, index_t q_index,
		     const RPoint& r1, index_t r1_index,
		     const RPoint& r2, index_t r2_index) {   
      if (unlikely((q_index == r1_index || q_index == r2_index) &&
		   param.Auto())){
	return;
      }
      if (r1_index == r2_index){
	return;
      }
      Vector dist;
      dist.Init(3);
      if (param.cartesian_){
	dist[0] = la::DistanceSqEuclidean(q.pos_, r1.pos_);
	dist[1] = la::DistanceSqEuclidean(q.pos_, r2.pos_);
	dist[2] = la::DistanceSqEuclidean(r1.pos_, r2.pos_);
      } else {
	if (param.redshift_){
	  dist[0] = mtrc::RedShiftDistSq(q.pos_, r1.pos_,
					 param.redshift_val_);
	  dist[1] = mtrc::RedShiftDistSq(q.pos_, r2.pos_,
					 param.redshift_val_);
	  dist[2] = mtrc::RedShiftDistSq(r1.pos_, r2.pos_,
					 param.redshift_val_);
	} else {
	  dist[0] = mtrc::SphereDistSq(q.pos_, r1.pos_);
	  dist[1] = mtrc::SphereDistSq(q.pos_, r2.pos_);
	  dist[2] = mtrc::SphereDistSq(r1.pos_, r2.pos_);
	}
      }
      local_three_.Add(dist);      
    }

  
    /** pass back the accumulated result into the query result
     */
    void FinishVisitingQueryPoint
      (const Param& param, const QPoint& q, index_t q_index,
       const RNode& r_node, const QSummaryResult& unapplied_summary_results,
       QResult* q_result, GlobalResult* global_result) {      
      global_result->two_point_.Merge(local_two_);   
      global_result->three_point_.Merge(local_three_);
      local_two_.Reset();
      local_three_.Reset();
    }
  };



 class TripleVisitor {
  public:        
    ThreePoint local_three_;
  private:
    
    
  public:
    void Init(const Param& param) {       
      local_three_.Init(param.bounds_);
    }
    
    /** apply single-tree based pruning by iterating over each query point
     */
    bool StartVisitingQueryPoint
      (const Param& param, const QPoint& q, index_t q_index,
       const RNode& r_node1, const RNode& r_node2, const Delta& delta,
       const QSummaryResult& unapplied_summary_results, QResult* q_result,
       GlobalResult* global_result) {
   
      double d, d1, d2, d3;     
      if (param.cartesian_){
	d1 = r_node1.bound().MinDistanceSq(q.pos_);
	d2 = r_node2.bound().MinDistanceSq(q.pos_);
	d3 = r_node1.bound().MinDistanceSq(r_node2.bound());
      } else {
	if (param.redshift_){
	  d1 = mtrc::MinRedShiftDistSq(r_node1.bound(), q.pos_,
				       param.redshift_val_);
	  d2 = mtrc::MinRedShiftDistSq(r_node2.bound(), q.pos_,
				       param.redshift_val_);
	  d3 = mtrc::MinRedShiftDistSq(r_node1.bound(), r_node2.bound(),
				       param.redshift_val_);
	} else {
	  d1 = mtrc::MinSphereDistSq(r_node1.bound(), q.pos_);
	  d2 = mtrc::MinSphereDistSq(r_node2.bound(), q.pos_);
	  d3 = mtrc::MinSphereDistSq(r_node1.bound(), r_node2.bound());
	}
      }
      d = max(d2, d3);
      d = max(d, d1);
      if (d > local_three_.Max()){
	return false;
      }
      
      // otherwise, we need to iterate over each reference point
     
      return true;
    }


    /** exhaustive computation between a query point and a reference point
     */
    void VisitTriple(const Param& param, const QPoint& q, index_t q_index,
		     const RPoint& r1, index_t r1_index,
		     const RPoint& r2, index_t r2_index) {   
      if (unlikely((q_index == r1_index || q_index == r2_index) &&
		   param.Auto())){
	return;
      }
      Vector dist;
      dist.Init(3);
      if (param.cartesian_){
	dist[0] = la::DistanceSqEuclidean(q.pos_, r1.pos_);
	dist[1] = la::DistanceSqEuclidean(q.pos_, r2.pos_);
	dist[2] = la::DistanceSqEuclidean(r1.pos_, r2.pos_);
      } else {
	if (param.redshift_){
	  dist[0] = mtrc::RedShiftDistSq(q.pos_, r1.pos_,
					 param.redshift_val_);
	  dist[1] = mtrc::RedShiftDistSq(q.pos_, r2.pos_,
					 param.redshift_val_);
	  dist[2] = mtrc::RedShiftDistSq(r1.pos_, r2.pos_,
					 param.redshift_val_);
	} else {
	  dist[0] = mtrc::SphereDistSq(q.pos_, r1.pos_);
	  dist[1] = mtrc::SphereDistSq(q.pos_, r2.pos_);
	  dist[2] = mtrc::SphereDistSq(r1.pos_, r2.pos_);
	}
      }
      local_three_.Add(dist);      
    }

  
    /** pass back the accumulated result into the query result
     */
    void FinishVisitingQueryPoint
      (const Param& param, const QPoint& q, index_t q_index,
       const RNode& r_node1, const RNode& rnode2, 
       const QSummaryResult& unapplied_summary_results,
       QResult* q_result, GlobalResult* global_result) {      
      global_result->three_point_.Merge(local_three_);   
      local_three_.Reset();
    }
  };






 class Algorithm {  

  public:

    /**
     * Calculates a delta....
     *
     * - If this returns true, delta is calculated, and global_result is
     * updated.  q_postponed is not touched.
     * - If this returns false, delta is not touched.
     */
    static bool ConsiderPairIntrinsic(const Param& param,
				      const QNode& q_node,
				      const RNode& r_node,
				      const Delta& parent_delta,
				      Delta* delta,
				      GlobalResult* global_result,
				      QPostponed* q_postponed) {
      double dmin;
      if (param.cartesian_){
	dmin = q_node.bound().MinDistanceSq(r_node.bound());
      } else{ 
	if (param.redshift_){
	  dmin = mtrc::MinRedShiftDistSq(q_node.bound(), r_node.bound(),
					 param.redshift_val_);  
	} else {      
	  dmin = mtrc::MinSphereDistSq(q_node.bound(), r_node.bound(),
				       param.redshift_val_);	
	}
      }
      if(dmin > global_result->two_point_.Max() &&
	 dmin > global_result->three_point_.Max()){
	return false;
      }      
      return true;      	
    }

    static bool ConsiderTripleIntrinsic(const Param& param,
					const QNode& q_node,
					const RNode& r_node1,
					const RNode& r_node2,
				      const Delta& parent_delta,
				      Delta* delta,
				      GlobalResult* global_result,
				      QPostponed* q_postponed) {      
      double dij, djk, dki;
      if (param.cartesian_){
	dij = q_node.bound().MinDistanceSq(r_node1.bound());
	dki = q_node.bound().MinDistanceSq(r_node2.bound());
	djk = r_node2.bound().MinDistanceSq(r_node1.bound());
      } else {
	if (param.redshift_){
	  dij = mtrc::MinRedShiftDistSq(q_node.bound(), r_node1.bound(),
					param.redshift_val_);  
	  dki = mtrc::MinRedShiftDistSq(q_node.bound(), r_node2.bound(),
					param.redshift_val_);  
	  djk = mtrc::MinRedShiftDistSq(r_node2.bound(), r_node1.bound(),
					param.redshift_val_);  
	} else {
	  dij = mtrc::MinSphereDistSq(q_node.bound(), r_node1.bound());  
	  dki = mtrc::MinSphereDistSq(q_node.bound(), r_node2.bound());  
	  djk = mtrc::MinSphereDistSq(r_node2.bound(), r_node1.bound()); 
	}	
      }
      if (max(max(dki, djk), dij) > global_result->three_point_.Max()){
	return false;
      }       
      return true;
    }
   
    static bool ConsiderPairExtrinsic(const Param& param,const QNode& q_node,
				      const RNode& r_node, const Delta& delta,
				      const QSummaryResult& q_summary_result,
				      const GlobalResult& global_result,
				      QPostponed* q_postponed) {    
      return true;      
    }


    static bool ConsiderTripleExtrinsic(const Param& param,const QNode& q_node,
					const RNode& r_node1, 
					const RNode& r_node2,
					const Delta& delta,
					const QSummaryResult& q_summary_result,
					const GlobalResult& global_result,
					QPostponed* q_postponed) {         
      return true;      
    }




    /**
     * Termination prune does not apply in KDE since all reference points
     * have to be considered...
     */
    static bool ConsiderQueryTermination
      (const Param& param, const QNode& q_node,
       const QSummaryResult& q_summary_result,
       const GlobalResult& global_result, QPostponed* q_postponed) {
      
      return true;
    }

    /**
     * Computes a heuristic for how early a computation should occur
     * -- smaller values are earlier.
     */
    static double Heuristic(const Param& param, const QNode& q_node,
			    const RNode& r_node, const Delta& delta) {
      return 1.0;
    }  

    static double Heuristic(const Param& param, const QNode& q_node,
			    const RNode& r_node1, const RNode& r_node2, 
			    const Delta& delta) {   
      return 1.0;
    }

 };

  // functions
 
 void ComputeNaive(){
   // create cache array for the distriuted caches storing the query
   // reference points and query results
   CacheArray<QPoint> q_points_cache_array;
   CacheArray<RPoint> r_points_cache_array;
   q_points_cache_array.Init(q_points_cache_, BlockDevice::M_READ);
   r_points_cache_array.Init(r_points_cache_, BlockDevice::M_READ);
   
   TwoPoint naive_two;
   naive_two.Init(parameters_.bounds_);
   ThreePoint naive_three;
   naive_three.Init(parameters_.bounds_);

   index_t q_end = (q_tree_->root()).end();
   index_t r_end = (r_tree_->root()).end();
   for(index_t q = (q_tree_->root()).begin(); q < q_end; q++) {
     
     CacheRead<QPoint> q_point(&q_points_cache_array, q);
     
     for(index_t r = (r_tree_->root()).begin(); r < r_end; r++) {
       if (likely((r != q) || !parameters_.Auto())){
	 CacheRead<RPoint> r_point(&r_points_cache_array, r);
	 
	 // compute pairwise and add contribution
	 double distance_sq;
	 if (parameters_.cartesian_){
	   distance_sq = la::DistanceSqEuclidean(q_point->vec(), 
						 r_point->vec());
	 } else {
	   if (parameters_.redshift_){
	     distance_sq = mtrc::RedShiftDistSq(q_point->vec(),
						r_point->vec(),
						param.redshift_val_);	   
	   } else {
	     distance_sq = mtrc::SphereDistSq(q_point->vec(), r_point->vec());
	   }
	 }       
	 // Add to correlation function
	 naive_two.Add(distance_sq);  

	 for (index_t r2 = (r_tree_->root()).begin(); r2 < r_end; r2++){
	   if (likely(((r2 != q) || !parameters_.Auto()) && r2 != r)){
	     CacheRead<RPoint> r2_point(&r_points_cache_array, r2);
	     Vector dist;
	     dist.Init(3);
	     if (parameters_.cartesian_){
	       dist[0]=la::DistanceSqEuclidean(q_point->vec(),r_point->vec());
	       dist[1]=la::DistanceSqEuclidean(q_point->vec(),r2_point->vec());
	       dist[2]=la::DistanceSqEuclidean(r_point->vec(),r2_point->vec());
	     } else {
	       if (parameters_.redshift_){
		 dist[0]=mtrc::RedShiftDistSq(q_point->vec(), r_point->vec(),
					      param.redshift_val_);
		 dist[1]=mtrc::RedShiftDistSq(q_point->vec(), r2_point->vec(),
					      param.redshift_val_);
		 dist[2]=mtrc::RedShiftDistSq(r_point->vec(), r2_point->vec(),
					      param.redshift_val_);
	       } else {
		 dist[0] = mtrc::SphereDistSq(q_point->vec(), r_point->vec());
		 dist[1] = mtrc::SphereDistSq(q_point->vec(), r2_point->vec());
		 dist[2] = mtrc::SphereDistSq(r_point->vec(), r2_point->vec());
	       }
	     }       
	     naive_three.Add(dist);
	   }   
	 }

       }

     } // finish looping over each reference point     
   }  // finish looping over each query point
   global_result_.Init(parameters_);
   global_result_.two_point_.Merge(naive_two);     
   naive_two.Reset();
   global_result_.three_point_.Merge(naive_three);     
   naive_three.Reset();
 }
 
 
 
 /** KDE computation using THOR */
 void Compute(datanode *module) {
   int do_naive = fx_param_int(module, "naive", 0);
   
   
   printf("Starting 2-Point Correlation...\n");
   fx_timer_start(module, "two_point");  
   
   if (do_naive){
     printf("Performing Naive Computation \n");
     ComputeNaive();
   } else {   
     q_results_.StartSync();
     q_results_.WaitSync();
     printf("Performing Dual-Tree Computation using THOR \n");
     thor::RpcDualTree<Thor3PC, ThreeTreeDepthFirst<Thor3PC> >
       (fx_submodule(module, "gnp"), GNP_CHANNEL,
	parameters_, q_tree_, r_tree_, &q_results_, &global_result_);
     q_results_.StartSync();
     q_results_.WaitSync();
   }
   fx_timer_stop(module, "two_point");
   printf("2-Point Correlation completed...\n");    
   global_result_.Postprocess(parameters_);
 
 }
 
 
 
 /** read datasets, build trees */    
 void Init(datanode *module) {
   
   // I don't quite understand what these mean, since I copied and pasted
   // from an example code.
   double results_megs = fx_param_double(module, "results/megs", 1000);
   
   // rpc::Init();    
   if (!rpc::is_root()) {
     //fx_silence();
   }
   
   // initialize parameter set    
   parameters_.Init(module);
   
   // read reference dataset
   // "data" gives name of input file
   fx_timer_start(module, "read_datasets");
   r_points_cache_ = new DistributedCache();
   parameters_.reference_count_ = 
     thor::ReadPoints<RPoint>(parameters_, DATA_CHANNEL + 0, DATA_CHANNEL + 1,
			      fx_submodule(module, "data"),
			      r_points_cache_);
   
   // read the query dataset if present
   if(fx_param_exists(module, "query")) {
     q_points_cache_ = new DistributedCache();
     parameters_.query_count_ = thor::ReadPoints<QPoint>
       (parameters_, DATA_CHANNEL + 2, DATA_CHANNEL + 3,
	fx_submodule(module, "query"), q_points_cache_);
     parameters_.SetAuto();
   } 
   else {
     q_points_cache_ = r_points_cache_;
     parameters_.query_count_ = parameters_.reference_count_;
   }
   fx_timer_stop(module, "read_datasets");
   
   Thor3PCPoint default_point;
   CacheArray<Thor3PCPoint>::GetDefaultElement(r_points_cache_, 
					       &default_point);   
   parameters_.FinalizeInit(module, default_point.vec().length());
   
   
   // construct trees
   fx_timer_start(module, "tree_construction");
   r_tree_ = new ThorTree<Param, RPoint, RNode>();
   thor::CreateKdTree<RPoint, RNode>(parameters_, DATA_CHANNEL + 4, 
				     DATA_CHANNEL + 5,
				     fx_submodule(module, "r_tree"),
				     parameters_.reference_count_, 
				     r_points_cache_, r_tree_);
   if (fx_param_exists(module, "query")) {
     q_tree_ = new ThorTree<Param, QPoint, QNode>();
     thor::CreateKdTree<QPoint, QNode>
       (parameters_, DATA_CHANNEL + 6, DATA_CHANNEL + 7, 
	fx_submodule(module, "q_tree"), parameters_.query_count_,
	q_points_cache_, q_tree_);
   } 
   else {
     q_tree_ = r_tree_;
   }
   fx_timer_stop(module, "tree_construction");
   
   // set up the cache holding query results
   QResult default_result;
   default_result.Init(parameters_);
   q_tree_->CreateResultCache(Q_RESULTS_CHANNEL, default_result,
			      results_megs, &q_results_);    
 }
 
 
 void OutputResults(Vector& counts_out){
   global_result_.two_point_.WriteResult(counts_out);
 }
 
 void GetBins(Vector& bins_out){
   parameters_.GetBins(bins_out);
 }

 void OutputResults3(ArrayList<ArrayList<Vector> >& counts_out){
   global_result_.three_point_.WriteResult(counts_out);
 }
 
 
 
 /** distributed cache for storing query results */
 DistributedCache q_results_;
 
 /** distributed cache for query points */
 DistributedCache *q_points_cache_;
 
 /** thor tree on query points */
 ThorTree<Param, QPoint, QNode> *q_tree_;
 
 /** distributed cache for reference points */
 DistributedCache *r_points_cache_;
 
 /** thor tree on reference points */
 ThorTree<Param, RPoint, RNode> *r_tree_;
 
 /** global parameter collection */
 Param parameters_;
 
 /** global results */
 GlobalResult global_result_;
 
 /** data channel */
 static const int DATA_CHANNEL = 110;
 
 /** query results channel */
 static const int Q_RESULTS_CHANNEL = 120;
 
 /** GNP channel ? */
 static const int GNP_CHANNEL = 200;
 
};

#endif
