/**

 */

#ifndef THOR_MD_H
#define THOR_MD_H

#include <fastlib/fastlib.h>
#include "fastlib/thor/thor.h"
#include "dfs3.h"
#include "two_body.h"
#include "three_body.h"
#include "periodic_tree.h"

/**
 * THOR-based Molecular Dynamics
 */

const fx_entry_doc module_entries[] = {
  {"bc", FX_PARAM, FX_INT, NULL,
   "Specifies boundary condition, 0 is free bound, 1 is periodic. \n"}, 
  {"prune_type", FX_PARAM, FX_INT, NULL,
   "Specifies pruning criterion. 0 is cutoff distance, 1 and 2 bound"
   "absolute error in potential and force, respectively. \n"}, 
  {"prune_val", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies value for pruning criterion."},
  {"pot_2", FX_REQUIRED, FX_STR, NULL,
   "Name of file specifying powers and signs for two-body potential"},
  {"lx", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies width of box in x-direction, if using periodic coordinates."},
  {"ly", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies width of box in y-direction. If not supplied, lx will be used."},
  {"lz", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies width of box in z-direction. If not supplied, lx will be used."},
  {"data", FX_PARAM, FX_STR, NULL,
   "Input file with positions, velocities, potential coefficients"},
   FX_ENTRY_DOC_DONE  
};

const fx_module_doc param_doc = {
  module_entries, NULL, 
  "Parameters of Simulated System \n"
};



class ThorMD {
  
  static const int FREE = 0;
  static const int PERIODIC = 1;
  static const int FIXED = 2;
  static const int CUTOFF = 0;
  static const int POTENTIAL = 1;
  static const int FORCE = 2;

 public:
  
  /** the bounding type which is required by THOR */
  typedef DHrectBound<2> Bound;
  

  class ThorMdStat;
  class ThorMdPoint;

  typedef ThorMdPoint QPoint;
  typedef ThorMdPoint RPoint;
  
  
  /** query stat */
  typedef ThorMdStat QStat;
  
  /** reference stat */
  typedef ThorMdStat RStat;

  /** query node */
  typedef ThorNode<Bound, QStat> QNode;

  /** reference node */
  typedef ThorNode<Bound, RStat> RNode;


  /** parameter class */
  class Param {
  public:

    // Powers for potentials are stored in param. Coefficients
    // are stored in point / node stats.
    TwoBody<QNode, RNode, ThorMdPoint> potential_;
    ThreeBody<QNode, RNode, ThorMdPoint> axilrod_;
    Vector box_size_;
    index_t query_count_;
    index_t reference_count_;
    int bound_type_, prune_type_;
    double prune_value_, prune_value2_;
    bool no_three_body_;

    OT_DEF(Param) {
      OT_MY_OBJECT(axilrod_);
      OT_MY_OBJECT(potential_);      
      OT_MY_OBJECT(box_size_);
      OT_MY_OBJECT(query_count_);
      OT_MY_OBJECT(bound_type_);
      OT_MY_OBJECT(prune_value_);
      OT_MY_OBJECT(prune_value2_);
      OT_MY_OBJECT(prune_type_);     
      OT_MY_OBJECT(no_three_body_);
    }
  public:
    
    /**
     * Initializes parameters from a data node (Req THOR).
     */
    void Init(datanode *module) {
      // Bounding Box
      bound_type_ = fx_param_int(module, "bc", PERIODIC);
      if (bound_type_ == PERIODIC){
	box_size_.Init(3);
	box_size_[0] = fx_param_double(module, "lx", 60);
	box_size_[1] = fx_param_double(module, "ly", box_size_[0]);
	box_size_[2] = fx_param_double(module, "lz", box_size_[0]);    
      } else {
	box_size_.Init(0);
      }
    
      // Pruning Criteria
      prune_type_ = fx_param_int(module, "prune_type", FORCE);
      if (prune_type_ == CUTOFF){
	prune_value_ = fx_param_double(module, "prune_val", 12.0);
	prune_value2_ = pow(prune_value_, 4)*pow(2.7, 5);
	prune_value2_ = pow(10*prune_value2_, 1.0/4.5);
	prune_value_ = prune_value_*prune_value_;
      }   else {      
	prune_value_ = fx_param_double(module, "prune_val", 1.0e-3);
      }

      // Two-body Potential
      Matrix potential_params;
      const char* fp_pot_param;
      fp_pot_param = fx_param_str_req(module, "pot_2");
      data::Load(fp_pot_param, &potential_params);
      potential_.Init(potential_params);
      
      // Three-body Potential
      axilrod_.Init(box_size_);
      no_three_body_ = fx_param_bool(module, "no_three", 0);
    }
    
    void FinalizeInit(datanode *module, int dimension) {
         
    }
  };
  
  /** 
   * the type of each KDE point - this assumes that each query and
   * each reference point is appended with a weight.
   */
  class ThorMdPoint {
  public:
    
    /** the point's position */
    Vector pos_, vel_, crossing_;
    Vector coefs_, axilrod_;
    double mass_;
    index_t old_index_;
    OT_DEF(ThorMdPoint) {
      OT_MY_OBJECT(coefs_);
      OT_MY_OBJECT(axilrod_);
      OT_MY_OBJECT(crossing_);
      OT_MY_OBJECT(pos_);
      OT_MY_OBJECT(vel_);
      OT_MY_OBJECT(mass_);
    }
    
  public:
    
    /** getters for the vector so that the tree-builder can access it */   
    const Vector& vec() const { return pos_; }
    Vector& vec() { return pos_; }
    
    /** initializes all memory for a point */
    void Init(const Param& param, const DatasetInfo& schema) {
      pos_.Init(3);
      vel_.Init(3);
      axilrod_.Init(2);     
      coefs_.Init( (param.potential_).n_terms());      
      if (param.bound_type_ == PERIODIC){
	crossing_.Init(3);
	crossing_.SetZero();
      } else {
	crossing_.Init(0);
      }
    }
    
    /** 
     * sets contents assuming all space has been allocated.
     * Any attempt to allocate memory here will lead to a core dump.
     */
    void Set(const Param& param, index_t index, Vector& data) {
      int i;      
      for (i = 0; i < 3; i++){
	pos_[i] = data[i];
	vel_[i] = data[i+4];
      }
      mass_ = data[3];  
      for (i = 0; i < coefs_.length(); i++){
	coefs_[i] = data[i+7];
      }
      for (i = data.length()-2; i < data.length(); i++){
	axilrod_[i-data.length() +2] = data[i];
      }
      old_index_ = index;
    }    

    void Accelerate(const Vector& acceleration_in, double time_step) {    
      /*
      if (old_index_ == 7371){
	printf("Acc: %f %f %f \n", acceleration_in[0], acceleration_in[1],  acceleration_in[2]);
	printf("Vel: %f %f %f \n", vel_[0], vel_[1], vel_[2]);
	printf("Pos: %f %f %f \n \n", pos_[0], pos_[1], pos_[2]);
      }
      */
      la::AddExpert(time_step, acceleration_in, &vel_);      
      la::AddExpert(time_step, vel_, &pos_);
    }    

    void MapBack(const Vector& box_size){
      for (int i = 0; i < 3; i++){
	double cross = floor(pos_[i] / box_size[i] + 0.5)*box_size[i];
	pos_[i] = pos_[i] - cross;
	crossing_[i] = crossing_[i] + cross;
      }
    }

    int GetPositionVector(Vector *point) const{
      if (crossing_.length() == 3){
	la::AddInit(crossing_, pos_, point);
      } else {
	point->Copy(pos_);	
      }
      return old_index_;
    }


    int GetFullVector(Vector *point) const{
      int n_data = 9 + coefs_.length();
      point->Init(n_data);
      for (int i = 0; i < 3; i++){
	(*point)[i] = pos_[i];
	(*point)[i+4] = vel_[i];	
      }
      (*point)[3] = mass_;
      for (int i = 0; i < coefs_.length(); i++){
	(*point)[i+7] = coefs_[i];
      }
      (*point)[n_data-2] = axilrod_[0];
      (*point)[n_data-1] = axilrod_[1];
      return old_index_;
    }

    void ScaleVelocity(double ratio){
      la::Scale(ratio, &vel_);
    }

  };


  
  /**
   * Per-node bottom-up statistic for both queries and references.
   *
   * The statistic must be commutative and associative, thus bottom-up
   * computable.
   *
   */
  class ThorMdStat {
    
  public:
    
    Vector centroid_;   
    // Coefs stores coefficients for two-body potential, 
    // axilrod stores coefficients for AT potential.
    Vector coefs_, axilrod_;
    double mass_;

    OT_DEF(ThorMdStat) {
      OT_MY_OBJECT(centroid_);
      OT_MY_OBJECT(coefs_);
      OT_MY_OBJECT(axilrod_);
      OT_MY_OBJECT(mass_);
    }
    
    /**
     * Initialize to a default zero value, as if no data is seen (Req THOR).
     *
     * This is the only method in which memory allocation can occur.
     */
  public:
    void Init(const Param& param) {
      centroid_.Init(3);
      centroid_.SetZero();
      coefs_.Init(param.potential_.n_terms()); // get value from param
      coefs_.SetZero();
      axilrod_.Init(2);
      axilrod_.SetZero();
      mass_ = 0.0;
    }
    
    /**
     * Accumulate data from a single point (Req THOR).
     */
    void Accumulate(const Param& param, const ThorMdPoint& point) {
      la::AddTo(point.axilrod_, &axilrod_);
      la::AddTo(point.coefs_, &coefs_);
   
      mass_ = mass_ + point.mass_;
      la::AddExpert(point.mass_, point.pos_, &centroid_);
    }
    
    /**
     * Accumulate data from one of your children (Req THOR).
     */
    void Accumulate(const Param& param, const ThorMdStat& child_stat, 
		    const Bound& bound, index_t child_n_points) {
      la::AddTo(child_stat.axilrod_, &axilrod_);
      la::AddTo(child_stat.coefs_, &coefs_);
      mass_ = mass_ + child_stat.mass_;
      la::AddExpert(child_stat.mass_, child_stat.centroid_, &centroid_);
    }
    
    /**
     * Finish accumulating data; for instance, for mean, divide by the
     * number of points.
     */
    void Postprocess(const Param& param, const Bound& bound, index_t n) {
      la::Scale(1.0 / mass_, &centroid_);      
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
  // This will store node-node velocities, if we're using
  // force or potential bounding 
  class QPostponed {
  public:

    Vector acceleration_;
    double error_budget_, triples_left_;

    OT_DEF_BASIC(QPostponed) {
      OT_MY_OBJECT(acceleration_);
      OT_MY_OBJECT(error_budget_);
      OT_MY_OBJECT(triples_left_);
    }

  public:
    
    /** initialize postponed information to zero */
    void Init(const Param& param) {
      triples_left_ = param.reference_count_*(param.reference_count_-1)/2;
      if (param.prune_type_ == CUTOFF){
      } else {
	error_budget_ = param.prune_value_;
      }
      acceleration_.Init(3);
      acceleration_.SetZero();
    }

    void Reset(const Param& param) {
     
    }

    /** accumulate postponed information passed down from above */
    void ApplyPostponed(const Param& param, const QPostponed& other) {
      la::AddTo(other.acceleration_, &acceleration_);    
      // Merge error statistics
    }
  };

  /** individual query result */
  class QResult {
  public:
    Vector acceleration_, old_acceleration_;
    double error_budget_, triples_left_;

    OT_DEF_BASIC(QResult) {
      OT_MY_OBJECT(acceleration_);
      OT_MY_OBJECT(old_acceleration_);
      OT_MY_OBJECT(error_budget_);
      OT_MY_OBJECT(triples_left_);
    }

  public:
    void Init(const Param& param) {      
      acceleration_.Init(3);
      acceleration_.SetZero();
      old_acceleration_.Init(3);
      old_acceleration_.SetZero();
      if (param.prune_type_ != CUTOFF){
	error_budget_ = param.prune_value_;
      }
      triples_left_ = param.reference_count_;
      triples_left_ = triples_left_*(triples_left_-1)/2;
    }

    void Seed(const Param& param, const QPoint& q) {
    }

    void UpdateError(double error_used, double trips_used){
      error_budget_ = error_budget_ - error_used;
      triples_left_ = triples_left_ - trips_used;
    }

    double ErrorRate(){
      return error_budget_ / triples_left_;
    }

    // Apply accelration to query point, and reset velocity.
    void Postprocess(const Param& param, const QPoint& q, index_t q_index,
		     const RNode& r_root) {
      old_acceleration_.CopyValues(acceleration_);
      acceleration_.SetZero();
    }   

    void AddVelocity(const Vector& acceleration_in){
      la::AddTo(acceleration_in, &acceleration_);
    }

    /** apply left over postponed contributions */
    void ApplyPostponed(const Param& param, const QPostponed& postponed,
			const QPoint& q, index_t q_index) {
      la::AddTo(postponed.acceleration_, &acceleration_);
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
    
    double virial_, temperature_, old_temp_, pressure_; 
    OT_DEF_BASIC(GlobalResult) {
      OT_MY_OBJECT(virial_);
      OT_MY_OBJECT(temperature_);     
      OT_MY_OBJECT(old_temp_);
      OT_MY_OBJECT(pressure_);    
    }



  public:
    void Init(const Param& param) {     
      temperature_ = 0;
      virial_ = 0;     
    }
    void Accumulate(const Param& param, const GlobalResult& other) {
      temperature_ = temperature_ + other.temperature_;
      virial_ = virial_ + other.virial_;
    }
    void ApplyDelta(const Param& param, const Delta& delta) {}
    void UndoDelta(const Param& param, const Delta& delta) {}

    void Postprocess(const Param& param) {
      if (param.box_size_.length() == 3){
	double volume = param.box_size_[0]*param.box_size_[1]*
	  param.box_size_[2];
	pressure_ = (temperature_ + virial_) / (3*volume);
      } else {
	pressure_ = 0;	
      }
      old_temp_ = temperature_ / (3*param.query_count_); 
      
      temperature_ = 0;
      virial_ = 0;   
    }

    void Report(const Param& param, datanode *datanode) {
    }
    void ApplyResult(const Param& param, const QPoint& q_point, index_t q_i,
		     const QResult& result) {
      Vector vel_;
      la::AddInit(result.old_acceleration_, q_point.vel_, &vel_);      
      temperature_ = temperature_ + q_point.mass_*la::Dot(vel_, vel_);
      virial_ = virial_ + la::Dot(q_point.pos_, result.acceleration_)*
	q_point.mass_;
    }    

  };
  
  /**
   * Abstract out the inner loop in a way that allows temporary variables
   * to be register-allocated.
   */
  class PairVisitor {
  public:
    // Velocity of q resulting from reference node
    Vector acceleration_;  
  private:
    
    
  public:
    void Init(const Param& param) {
      acceleration_.Init(3);
      acceleration_.SetZero();  
    }
    
    /** apply single-tree based pruning by iterating over each query point
     */
    bool StartVisitingQueryPoint
      (const Param& param, const QPoint& q, index_t q_index,
       const RNode& r_node, const Delta& delta,
       const QSummaryResult& unapplied_summary_results, QResult* q_result,
       GlobalResult* global_result) {         
      acceleration_.SetZero();
      // if we can prune the entire reference node for the given query point,
      // then we are done
      double bound;
      if (param.prune_type_ == CUTOFF){
	// Check distances
	if (param.bound_type_ == PERIODIC){
	  bound = prdc::MinDistanceSq(r_node.bound(), q.pos_,
					  param.box_size_);
	} else {
	  bound = r_node.bound().MinDistanceSq(q.pos_);
	}
	if (bound > param.prune_value_) {
	  return false;
	}
      } else {
	int n_trips = r_node.count()*(r_node.count()-1)/2;
	if (param.prune_type_ == POTENTIAL){
	  bound = param.potential_.PotentialRange(q, r_node, param.box_size_); 
	  // bound += param.axilrod_.PotentialRange(q, r_node); 
	} else {
	  bound = param.potential_.ForceRange(q, r_node, param.box_size_); 
	  // bound += param.axilrod_.ForceRange(q, r_node);
	}
	if (bound < 0){
	  Vector force;	
	  param.potential_.ForceVector(q, r_node, param.box_size_, &force);
	  q_result->AddVelocity(force);
	  q_result->UpdateError(bound, n_trips);
	  return false;
	}
      }
      
      // otherwise, we need to iterate over each reference point     
      return true;
    }


    /** exhaustive computation between a query point and a reference point
     */
    void VisitPair(const Param& param, const QPoint& q, index_t q_index,
		   const RPoint& r, index_t r_index) {
     if (unlikely(q_index == r_index)){
	return;
     }    
     Vector force;     
     if (param.prune_type_ == CUTOFF){
       param.potential_.ForceVector(q, r, param.box_size_,param.prune_value_, &force);      
     } else {
       param.potential_.ForceVector(q, r, param.box_size_, &force);        
     }
     la::AddTo(force, &acceleration_);
    }
    
    void VisitTriple(const Param& param, const QPoint& q, index_t q_index,
		     const RPoint& r1, index_t r1_index,
		     const RPoint& r2, index_t r2_index) {
      if (unlikely((q_index == r1_index) || (q_index == r2_index))){
	return;
      }            
      Vector force;
      if (param.prune_type_ == CUTOFF){	  
	param.axilrod_.ForceVector(q, r1, r2, param.prune_value_, 
				   param.prune_value2_, &force);
	/*
	  if (q.old_index_ == 7371){  
	    FILE* stuff;
	    stuff = fopen("stuff.dat", "a+");
	    fprintf(stuff, "%d %d %f %f %f \n", r1.old_index_, r2.old_index_, 
		    force[0], force[1], force[2]);
	  }
	*/
      } else {
	param.axilrod_.ForceVector(q, r1, r2, &force);
      }
      la::AddTo(force, &acceleration_);
    }
    

    /** pass back the accumulated result into the query result
     */
    void FinishVisitingQueryPoint
      (const Param& param, const QPoint& q, index_t q_index,
       const RNode& r_node, const QSummaryResult& unapplied_summary_results,
       QResult* q_result, GlobalResult* global_result) {
      q_result->AddVelocity(acceleration_);
      acceleration_.SetZero();     
    }
  };


 class TripleVisitor {

  public:
    
   // Acceleration for query from these refs
   Vector acceleration_;  
 private: 
  

 public:
    void Init(const Param& param) {
      acceleration_.Init(3);
      acceleration_.SetZero();     
    }    
  
    /** apply single-tree based pruning by iterating over each query point
     */
    bool StartVisitingQueryPoint
      (const Param& param, const QPoint& q, index_t q_index,
       const RNode& r_node1, const RNode& r_node2, const Delta& delta,
       const QSummaryResult& unapplied_summary_results, QResult* q_result,
       GlobalResult* global_result) {
      
        
      // if we can prune the entire reference node for the given query point,
      // then we are done     
      if (param.prune_type_ == CUTOFF){
	// Check distances
	double dij, djk, dki;
	if (param.bound_type_ == PERIODIC){
	  dij = prdc::MinDistanceSq(r_node2.bound(), q.pos_, param.box_size_);
	  dki = prdc::MinDistanceSq(r_node1.bound(), q.pos_, param.box_size_);
	  djk = prdc::MinDistanceSq(r_node2.bound(), r_node1.bound(), 
				    param.box_size_);
	} else {
	  dij = r_node2.bound().MinDistanceSq(q.pos_);
	  dki = r_node1.bound().MinDistanceSq(q.pos_);
	  djk = r_node2.bound().MinDistanceSq(r_node1.bound());
	}
	double c1, c2;
	c1 = param.prune_value_;
	c2 = param.prune_value2_;
	if ((dij > c1) || (djk > c1) || (dki > c1) || 
	    ((dij > c2) && (djk > c2) && (dki > c2))){
	  return false;
	}
      } else { 
	double bound; 
	int n_trips = r_node2.count()*r_node1.count();	
	if (param.prune_type_ == POTENTIAL){	 
	  //	  bound = param.axilrod_.PotentialRange(q, r_node1, r_node2); 
	} else {	 
	  //	  bound = param.axilrod_.ForceRange(q, r_node1, r_node2); 
	}
	if (bound/n_trips < q_result->ErrorRate()){
	  // Apply Force
	  Vector force;
	  param.axilrod_.ForceVector(q, r_node1, r_node2, &force);	
	  q_result->AddVelocity(force);
	  q_result->UpdateError(bound, n_trips);	  
	  return false;
	}
	return true;
      }
      
      
      
      // otherwise, we need to iterate over each reference point
     
      return true;
    }

    /** exhaustive computation between a query point and a reference point
     */    
    void VisitTriple(const Param& param, const QPoint& q, index_t q_index,
		     const RPoint& r1, index_t r1_index,
		     const RPoint& r2, index_t r2_index) {
      // We've screened for r1 != r2, but r1 or r1 might equal q.
      if (unlikely((q_index == r1_index) || (q_index == r2_index))){
	return;
      }   
      Vector force;
      if (param.prune_type_ == CUTOFF){
	param.axilrod_.ForceVector(q, r1, r2, param.prune_value_, param.prune_value2_, 
				   &force);
	  /*
	  if (q.old_index_ == 7371){	   
	    FILE* stuff;
	    stuff = fopen("stuff.dat", "a+");
	    fprintf(stuff, "%d %d %f %f %f \n", r1.old_index_, r2.old_index_, 
		    force[0], force[1], force[2]);
	    printf("%d %d %f %f %f \n", r1.old_index_, r2.old_index_, 
		    force[0], force[1], force[2]);
	  }
	  */
	
      } else {
	param.axilrod_.ForceVector(q, r1, r2, &force);
	la::AddTo(force, &acceleration_);
      }
    }
    

    /** pass back the accumulated result into the query result
     */
    void FinishVisitingQueryPoint
      (const Param& param, const QPoint& q, index_t q_index,
       const RNode& r_node1, const RNode& rnode2, 
       const QSummaryResult& unapplied_summary_results,
       QResult* q_result, GlobalResult* global_result) {
      q_result->AddVelocity(acceleration_);
      acceleration_.SetZero();    
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
      if (param.prune_type_ == CUTOFF){
	// compute distance bound between two nodes
	double dist;
	if (likely(param.bound_type_ == PERIODIC)){
	  dist = prdc::MinDistanceSq(q_node.bound(), r_node.bound(), 
					 param.box_size_);	
	} else {
	  dist = q_node.bound().MinDistanceSq(r_node.bound());
	}
	if (dist > param.prune_value_) {
	  return false;
	}
      } 
      return true;      	
    }

     /**
     * Calculates a delta....
     *
     * - If this returns true, delta is calculated, and global_result is
     * updated.  q_postponed is not touched.
     * - If this returns false, delta is not touched.
     */
    static bool ConsiderTripleIntrinsic(const Param& param,
					const QNode& q_node,
					const RNode& r_node1,
					const RNode& r_node2,
				      const Delta& parent_delta,
				      Delta* delta,
				      GlobalResult* global_result,
				      QPostponed* q_postponed) {
      
      // compute distance bound between two nodes
      if (param.no_three_body_){
	return false;
      }

      if (param.prune_type_ == CUTOFF) {
	double dij, djk, dki;
	if (likely(param.bound_type_ == PERIODIC)){
	  dij = prdc::MinDistanceSq(q_node.bound(), r_node1.bound(), 
				    param.box_size_);	 
	  dki = prdc::MinDistanceSq(q_node.bound(), r_node2.bound(), 
				    param.box_size_);	 
	  djk = prdc::MinDistanceSq(r_node2.bound(), r_node1.bound(), 
				    param.box_size_);	
	} else {
	  dij = q_node.bound().MinDistanceSq(r_node1.bound());
	  dki = q_node.bound().MinDistanceSq(r_node2.bound());
	  djk = r_node2.bound().MinDistanceSq(r_node1.bound());
	}
	double c1, c2;
	c1 = param.prune_value_;
	c2 = param.prune_value2_;
	if ((dij > c1 || djk > c1 || dki > c1) ||
	    ((dij > c2) && (djk > c2) && (dki > c2))){
	  return false;
	}
      } 
      return true;
    }

    /**
     * Prune based on the accumulated lower bound contribution and allocated
     * error
     */
    static bool ConsiderTripleExtrinsic(const Param& param,const QNode& q_node,
					const RNode& r_node1, 
					const RNode& r_node2,
					const Delta& delta,
					const QSummaryResult& q_summary_result,
					const GlobalResult& global_result,
					QPostponed* q_postponed) {     
      if (param.prune_type_ != CUTOFF){
	int n_trips;
	n_trips = r_node1.count()*r_node2.count();
	double bound = BIG_BAD_NUMBER;
	if (param.prune_type_ == POTENTIAL){
	  bound = param.axilrod_.PotentialRange(q_node, r_node1, r_node2);
	} else {
	  bound = param.axilrod_.ForceRange(q_node, r_node1, r_node2);  
	} 
	if (bound / n_trips < q_postponed->error_budget_ / 
	    q_postponed->triples_left_){
	  // Evaluate Force Here
	  Vector force;
	  param.axilrod_.ForceVector(q_node, r_node1, r_node2, &force);
	  q_postponed->error_budget_ = q_postponed->error_budget_ - bound;
	  q_postponed->triples_left_ = q_postponed->triples_left_ - n_trips;
	  return false;
	}
      }       
      return true;      
    }

    static bool ConsiderPairExtrinsic(const Param& param,const QNode& q_node,
				      const RNode& r_node, const Delta& delta,
				      const QSummaryResult& q_summary_result,
				      const GlobalResult& global_result,
				      QPostponed* q_postponed) {
      if (param.prune_type_ != CUTOFF){
	int n_trips;
	double bound = BIG_BAD_NUMBER;
	n_trips = (r_node.count()*(r_node.count()-1)) / 2;
	if (param.prune_type_ == POTENTIAL){
	  bound = param.potential_.PotentialRange(q_node, r_node, 
						   param.box_size_);
	  //  bound = bound + param.axilrod_.PotentialRange(q_node, r_node);
	} else {
	  bound = param.potential_.ForceRange(q_node,r_node, param.box_size_);
	  //	  bound = bound + param.axilrod_.ForceRange(q_node, r_node);
	} 
	if (bound / n_trips < q_postponed->error_budget_ / 
	    q_postponed->triples_left_){
	  // Evaluate Force Here
	  Vector force;
	  param.potential_.ForceVector(q_node,r_node, param.box_size_, &force);
	  la::AddTo(force, &(q_postponed->acceleration_));
	  q_postponed->error_budget_ = q_postponed->error_budget_ - bound;
	  q_postponed->triples_left_ = q_postponed->triples_left_ - n_trips;
	  return false;
	}
      }       
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
      double dist;
      if (likely(param.bound_type_ == PERIODIC)){
	dist = prdc::MinDistanceSq(r_node.bound(), q_node.bound(),
				   param.box_size_);
      } else {
	dist = r_node.bound().MinDistanceSq(q_node.bound());
      }
      return dist;
    }

    static double Heuristic(const Param& param, const QNode& q_node,
			    const RNode& r_node1, const RNode& r_node2, 
			    const Delta& delta) {
      double dist;
      if (likely(param.bound_type_ == PERIODIC)){
	return 1;
	/*
	dist = 
	  prdc::MinDistanceSq(r_node1.bound(),q_node.bound(),param.box_size_)*
	  prdc::MinDistanceSq(r_node1.bound(),r_node2.bound(),param.box_size_)*
	  prdc::MinDistanceSq(q_node.bound(), r_node2.bound(),param.box_size_);
	*/
      } else{
	dist = r_node1.bound().MinDistanceSq(q_node.bound())*
	  r_node1.bound().MinDistanceSq(r_node2.bound())*
	  q_node.bound().MinDistanceSq(r_node2.bound());
      }
      return dist;
    }
  };

  // functions


  /** KDE computation using THOR */ 
  void Compute(datanode *module) {      
    q_results_.StartSync();
    q_results_.WaitSync();
    fx_timer_start(module, "dualtree md");  
    thor::RpcDualTree<ThorMD, ThreeTreeDepthFirst<ThorMD> >
      (fx_submodule(module, "gnp"), GNP_CHANNEL,
       parameters_, q_tree_, r_tree_, &q_results_, &global_result_);
    fx_timer_stop(module, "dualtree md");  
    global_result_.Postprocess(parameters_);  
    q_results_.StartSync();
    q_results_.WaitSync();
  }

  

  /** read datasets, build trees */    
  void Init(datanode *module) {

    chan_ = 8;
    // I don't quite understand what these mean, since I copied and pasted
    // from an example code.
    double results_megs = fx_param_double(module, "results/megs", 1000);

    // rpc::Init();    
    if (!rpc::is_root()) {
      //   fx_silence();
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
    } 
    else {
      q_points_cache_ = r_points_cache_;
      parameters_.query_count_ = parameters_.reference_count_;
    }
    fx_timer_stop(module, "read_datasets");
   
    ThorMdPoint default_point;
    CacheArray<ThorMdPoint>::GetDefaultElement(r_points_cache_, 
						&default_point);   
    parameters_.FinalizeInit(module, default_point.vec().length());

    
    // construct trees
    fx_timer_start(module, "tree_construction");
    r_tree_ = new ThorTree<Param, RPoint, RNode>();
    thor::CreateKdTree<RPoint, RNode>(parameters_, DATA_CHANNEL + 4, 
				      DATA_CHANNEL + chan_,
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

  void RebuildTree(datanode* module){  
    r_tree_->nodes().StartSync();   
    r_tree_->nodes().WaitSync();    
    delete &r_tree_->nodes();
    r_tree_->param().Renew();
    r_tree_->decomp().Renew();         
    thor::CreateKdTree<RPoint, RNode>(parameters_, DATA_CHANNEL + chan_, 
				      DATA_CHANNEL + 5,
				      fx_submodule(module, "r_tree"),
				      parameters_.reference_count_, 
				      r_points_cache_, r_tree_);
    if (chan_ == 9){
      chan_ = 8;
    } else {
      chan_ = 9;
    }
    q_tree_ = r_tree_;     
  }

  double GetDiffusion(Matrix& positions){
    double diff = 0;         
    if (rpc::is_root()){      
      CacheArray<QPoint> points_array;
      points_array.Init(q_points_cache_, BlockDevice::M_READ);
      CacheReadIter<QPoint> points_iter(&points_array, 0);
      for (index_t i = 0; i < parameters_.query_count_; i++,
	     points_iter.Next()){
	Vector point;
	int k = (*points_iter).GetPositionVector(&point);
	for (index_t j = 0; j < 3; j++){
	  diff = diff + (point[j] - positions.get(j, k))*
	    (point[j] - positions.get(j, k));
	}
      }
      diff = diff / parameters_.query_count_;
    }
    q_points_cache_->StartSync();
    q_points_cache_->WaitSync();   
    return diff;
  }

  void UpdatePoints(double time_step){   
    q_points_cache_->StartSync();
    if (rpc::is_root()){
      CacheArray<QResult> result_array;
      CacheArray<QPoint> points_array;
      result_array.Init(&q_results_, BlockDevice::M_READ);
      points_array.Init(q_points_cache_, BlockDevice::M_OVERWRITE);
      CacheReadIter<QResult> result_iter(&result_array, 0);
      CacheWriteIter<QPoint> points_iter(&points_array, 0);
      for (index_t i = 0; i < parameters_.query_count_; i++,
	     result_iter.Next(), points_iter.Next()) {
	(*points_iter).Accelerate((*result_iter).old_acceleration_, time_step);
	if (parameters_.bound_type_ ==PERIODIC){
	  (*points_iter).MapBack(parameters_.box_size_);
	}
      }
    }  
    q_points_cache_->WaitSync();    
  }
  
  void ScaleToTemperature(double ratio){

    if (rpc::is_root()){
      CacheArray<QPoint> points_array;   
      points_array.Init(q_points_cache_, BlockDevice::M_OVERWRITE);   
      CacheWriteIter<QPoint> points_iter(&points_array, 0);
      for (index_t i = 0; i < parameters_.query_count_; i++, 
	     points_iter.Next()) {
	(*points_iter).ScaleVelocity(ratio);    
      }   
    }       
    q_points_cache_->StartSync();
    q_points_cache_->WaitSync(); 
  }


  void Fin(){
    delete r_tree_;
    rpc::Done();
  }

  void TakeSnapshot(Matrix* out_positions){
    if (rpc::is_root()){
      out_positions->Init(3, parameters_.query_count_);
      CacheArray<QPoint> points_array;
      points_array.Init(q_points_cache_, BlockDevice::M_READ);
      CacheReadIter<QPoint> points_iter(&points_array, 0);
      for (index_t i = 0; i < parameters_.query_count_; i++,
	     points_iter.Next()){
	Vector out_point;
	int k = (*points_iter).GetPositionVector(&out_point);
	for (index_t j = 0; j < 3; j++){
	  out_positions->set(j, k, out_point[j]);
	}
      }
    } else {
      out_positions->Init(0,0);
    }
    q_points_cache_->StartSync();
    q_points_cache_->WaitSync();    
  }

  void GetFinalPositions(Matrix* out_positions){
    if (rpc::is_root()){
      out_positions->Init(parameters_.potential_.n_terms() + 9,
			  parameters_.query_count_);
      CacheArray<QPoint> points_array;
      points_array.Init(q_points_cache_, BlockDevice::M_READ);
      CacheReadIter<QPoint> points_iter(&points_array, 0);
      for (index_t i = 0; i < parameters_.query_count_; i++,
	     points_iter.Next()){
	Vector out_point;
	int k = (*points_iter).GetFullVector(&out_point);
	for (index_t j = 0; j < out_point.length(); j++){
	  out_positions->set(j, k, out_point[j]);
	}
      }
    } else {
      out_positions->Init(0,0);
    }
    q_points_cache_->StartSync();
    q_points_cache_->WaitSync();    
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

  int chan_;

  /** data channel */
  static const int DATA_CHANNEL = 110;

  /** query results channel */
  static const int Q_RESULTS_CHANNEL = 120;

  /** GNP channel ? */
  static const int GNP_CHANNEL = 200;

};

#endif
