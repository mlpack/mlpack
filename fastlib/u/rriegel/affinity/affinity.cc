
// TODO: Do not consider a point with itself

// OVERALL
// -  Get the code to something that looks like completion
// -  Think about splitting on both kids
// -  Rethink the guess system, who corrects who?
// -  I seriously doubt DFS is the best...

#include "base/common.h"
#include "xrun/xrun.h"
#include "linear/linear.h"
#include "linear/dmatrix.h"
#include "collections/arraylist.h"
#include "collections/heap.h"
#include "file/textmatrix.h"
#include "tree/spacetree.h"
#include "tree/bounds.h"
#include "tree/kdtree.h"

#include <cmath>
#include <cfloat>

// TODO: divide:  1/(bandwidth * sqrt(2pi))

#define EPSILON (1.0e-7)
#define PI 3.141592653589793238462643383279

struct AffinityStat {
  struct Max {
    double first;
    double second;
    index_t index;

    void Init() {
      first = -DBL_MAX;
      second = -DBL_MAX;
      index = 0;
    }
  };

  struct Sum {
    double val;

    void Init() {
      val = 0;
    }
  };

  double maxA_lower;
  double maxA_upper;
  double maxB_lower;
  double maxB_upper;
  double sum_lower;
  double sum_upper;
  double sum_lower_ex;
  double sum_upper_ex;

  int remaining;
  double spent;

  void Init() {
    maxA_lower = -DBL_MAX;
    maxA_upper = DBL_MAX;
    maxB_lower = -DBL_MAX;
    maxB_upper = DBL_MAX;
    sum_lower = 0;
    sum_upper = 0;
    sum_lower_ex = 0;
    sum_upper_ex = 0;
  }
  
  void Init(const DMatrix& dataset, index_t start, index_t count) {
    Init();
  }
  
  void Init(const DMatrix& dataset, index_t start, index_t count,
      const AffinityStat &left_stat, const AffinityStat &right_stat) {
    Init();
  }
};

typedef BinarySpaceTree<DHrectBound, DMatrix, AffinityStat> AffinityTree;

struct Max2Problem {
  double pref;
  bool update_B;
  DMatrix d_matrix;
  ArrayList<AffinityStat::Max> *maxes;

  uint64 pruned_pairs;
  uint64 pruned_area;

  Max2Problem() {}
  ~Max2Problem() {}

  void Init(double pref_in, const DMatrix& d_matrix_in, AffinityTree *d,
	    ArrayList<AffinityStat::Max> *maxes_in, bool update_B_in)
  {
    FL_DEBUG_ASSERT(d->count() == d_matrix_in.n_cols());

    pref = pref_in;
    update_B = update_B_in;
    d_matrix.Alias(d_matrix_in);

    maxes = maxes_in;
    ArrayList<AffinityStat::Max> &maxes_temp = *maxes_in;
    for (index_t i = 0; i < d_matrix.n_cols(); i++) {
      maxes_temp[i].Init();
    }

    pruned_pairs = 0;
    pruned_area = 0;
  }
  
  void Report() {
    xrun_metric_set("affinity_gnp", "pruned_pairs", "%Lu",
        (unsigned long long)(pruned_pairs));
    xrun_metric_set("affinity_gnp", "pruned_area", "%Lu",
        (unsigned long long)(pruned_area));
    xrun_metric_set("affinity_gnp", "area_per_prune", "%f",
	double(pruned_area) / pruned_pairs);
  }

  double UpperBoundSim(const AffinityTree *q, const AffinityTree *r) const {
    double dist = sqrt(q->bound().MinDistanceSqToBound(r->bound()));
    if (dist == 0) {
      return DBL_MAX;
    }

    double upper_bound = 1.0 / dist;
    index_t q_end = q->first() + q->count();
    index_t r_end = r->first() + r->count();

    if (q->first() < r_end && r->first() < q_end && pref > upper_bound) {
      upper_bound = pref;
    }

    return upper_bound;
  }

  double ExactSim(index_t q_i, index_t r_i) const {
    if (unlikely(q_i == r_i)) {
      return pref;
    }

    DVector q_vec, r_vec;
    d_matrix.MakeColumnVector(q_i, &q_vec);
    d_matrix.MakeColumnVector(r_i, &r_vec);

    double dist = sqrt(linear::DistanceSqEuclidean(q_vec,r_vec));
    if (unlikely(dist == 0)) {
      return DBL_MAX;
    }

    return 1.0 / dist;
  }
};

struct Max2NodePair {
  AffinityTree *q;
  AffinityTree *r;

  double upper_bound;

  Max2NodePair() {}
  ~Max2NodePair() {}

  Max2NodePair(AffinityTree *q_in, AffinityTree *r_in, Max2Problem *problem) {
    q = q_in;
    r = r_in;
    upper_bound = problem->UpperBoundSim(q, r);
  }
};

void max_2_base(AffinityTree *q, AffinityTree *r, Max2Problem *problem) {
  index_t q_end = q->first() + q->count();
  index_t r_end = r->first() + r->count();

  ArrayList<AffinityStat::Max> &maxes = *(problem->maxes);
  double max_lower = DBL_MAX;
  double max_upper = -DBL_MAX;

  for (index_t q_i = q->first(); q_i < q_end; q_i++) {

    double first_max = maxes[q_i].first;
    double second_max = maxes[q_i].second;
    int max_index = maxes[q_i].index;
    bool alt = false;

    for (index_t r_i = r->first(); r_i < r_end; r_i++) {

      double sim = problem->ExactSim(q_i, r_i);

      if (unlikely(sim > second_max)) {
	if (unlikely(sim > first_max)) {
	  second_max = first_max;
	  first_max = sim;
	  max_index = r_i;
	}
	else {
	  second_max = sim;
	}

	alt = true;
      }
    }

    if (unlikely(alt)) {
      maxes[q_i].first = first_max;
      maxes[q_i].second = second_max;
      maxes[q_i].index = max_index;
    }

    if (second_max < max_lower) {
      max_lower = second_max;
    }
    if (first_max > max_upper) {
      max_upper = first_max;
    }
  }

  if (problem->update_B) {
    q->statistic().maxB_lower = max_lower;
    q->statistic().maxB_upper = max_upper;
  }
  else {
    q->statistic().maxA_lower = max_lower;
    q->statistic().maxA_upper = max_upper;
  }
}

void max_2_body(Max2NodePair p, Max2Problem *problem) {
  AffinityTree *q = p.q, *r = p.r;

  if ((problem->update_B && p.upper_bound <= q->statistic().maxB_lower)
      || (!problem->update_B && p.upper_bound <= q->statistic().maxA_lower)) {
    problem->pruned_pairs++;
    problem->pruned_area += q->count() * r->count();
  }
  else {
    if (q->is_leaf() && r->is_leaf()) {
      max_2_base(q, r, problem);
    }
    else if (q->is_leaf()) {
      Max2NodePair pA(q, r->left(), problem);
      Max2NodePair pB(q, r->right(), problem);
      if (pA.upper_bound > pB.upper_bound) {
	max_2_body(pA, problem);
	max_2_body(pB, problem);
      }
      else {
	max_2_body(pB, problem);
	max_2_body(pA, problem);
      }
    }
    else if (r->is_leaf()) {
      max_2_body(Max2NodePair(q->left(), r, problem), problem);
      max_2_body(Max2NodePair(q->right(), r, problem), problem);

      if (problem->update_B) {
	q->statistic().maxB_lower =
	  min(q->left()->statistic().maxB_lower,
	      q->right()->statistic().maxB_lower);
	q->statistic().maxB_upper =
	  max(q->left()->statistic().maxB_upper,
	      q->right()->statistic().maxB_upper);
      }
      else {
	q->statistic().maxA_lower =
	  min(q->left()->statistic().maxA_lower,
	      q->right()->statistic().maxA_lower);
	q->statistic().maxA_upper =
	  max(q->left()->statistic().maxA_upper,
	      q->right()->statistic().maxA_upper);
      }
    }
    else {
      {
	Max2NodePair pA(q->left(), r->left(), problem);
	Max2NodePair pB(q->left(), r->right(), problem);
	if (pA.upper_bound > pB.upper_bound) {
	  max_2_body(pA, problem);
	  max_2_body(pB, problem);
	}
	else {
	  max_2_body(pB, problem);
	  max_2_body(pA, problem);
	}
      }

      {
	Max2NodePair pA(q->right(), r->left(), problem);
	Max2NodePair pB(q->right(), r->right(), problem);
	if (pA.upper_bound > pB.upper_bound) {
	  max_2_body(pA, problem);
	  max_2_body(pB, problem);
	}
	else {
	  max_2_body(pB, problem);
	  max_2_body(pA, problem);
	}
      }

      if (problem->update_B) {
	q->statistic().maxB_lower =
	  min(q->left()->statistic().maxB_lower,
	      q->right()->statistic().maxB_lower);
	q->statistic().maxB_upper =
	  max(q->left()->statistic().maxB_upper,
	      q->right()->statistic().maxB_upper);
      }
      else {
	q->statistic().maxA_lower =
	  min(q->left()->statistic().maxA_lower,
	      q->right()->statistic().maxA_lower);
	q->statistic().maxA_upper =
	  max(q->left()->statistic().maxA_upper,
	      q->right()->statistic().maxA_upper);
      }
    }
  }
}

void max_2(double pref, const DMatrix& d_matrix, AffinityTree *d,
	   ArrayList<AffinityStat::Max> *maxesA,
	   ArrayList<AffinityStat::Max> *maxesB, bool update_B)
{
  Max2Problem problem;

  if (update_B) {
    problem.Init(pref, d_matrix, d, maxesB, update_B);
  }
  else {
    problem.Init(pref, d_matrix, d, maxesA, update_B);
  }
  max_2_body(Max2NodePair(d, d, &problem), &problem);

  problem.Report();
}

void affinity_sum_reset(AffinityTree *d, int remaining)
{
  d->statistic().remaining = remaining;
  d->statistic().spent = 0;

  d->statistic().sum_lower = 0;
  d->statistic().sum_upper = 0;

  if (d->is_leaf()) {
    d->statistic().sum_lower_ex = 0;
    d->statistic().sum_upper_ex = 0;
  }
  else {
    affinity_sum_reset(d->left(), remaining);
    affinity_sum_reset(d->right(), remaining);
  }
}

struct AffinitySumNodePair;

struct AffinitySumProblem {
  double epsilon;
  double pref;
  bool update_B;
  DMatrix d_matrix;
  ArrayList<AffinityStat::Max> *maxes;
  ArrayList<AffinityStat::Sum> *sums;
  AffinitySumNodePair *head;
  AffinitySumNodePair *tail;

  uint64 pruned_pairs;
  uint64 pruned_area;

  AffinitySumProblem() {}
  ~AffinitySumProblem() {}

  void Init(double pref_in, const DMatrix& d_matrix_in, AffinityTree *d,
	    ArrayList<AffinityStat::Max> *maxes_in,
	    ArrayList<AffinityStat::Sum> *sums_in, bool update_B_in,
	    double epsilon_in)
  {
    FL_DEBUG_ASSERT(d->count() == d_matrix_in.n_cols());

    epsilon = epsilon_in;
    pref = pref_in;
    update_B = update_B_in;
    d_matrix.Alias(d_matrix_in);

    maxes = maxes_in;
    sums = sums_in;
    ArrayList<AffinityStat::Sum> &sums_temp = *sums_in;
    for (index_t i = 0; i < d_matrix.n_cols(); i++) {
      sums_temp[i].Init();
    }

    affinity_sum_reset(d, d->count());

    head = tail = NULL;

    pruned_pairs = 0;
    pruned_area = 0;
  }

  void Report() {
    xrun_metric_set("affinity_gnp", "pruned_pairs", "%Lu",
        (unsigned long long)(pruned_pairs));
    xrun_metric_set("affinity_gnp", "pruned_area", "%Lu",
        (unsigned long long)(pruned_area));
    xrun_metric_set("affinity_gnp", "area_per_prune", "%f",
	double(pruned_area) / pruned_pairs);
  }

  AffinitySumNodePair *Pop();
  void Push(AffinitySumNodePair *p);

  double UpperBound(const AffinityTree *q, const AffinityTree *r) const
  {
    double dist = sqrt(q->bound().MinDistanceSqToBound(r->bound()));
    if (dist == 0) {
      return DBL_MAX;
    }

    double upper_bound = 1.0 / dist;
    index_t q_end = q->first() + q->count();
    index_t r_end = r->first() + r->count();

    if (q->first() < r_end && r->first() < q_end && pref > upper_bound) {
      upper_bound = pref;
    }

    if (update_B) {
      return r->count() * max(0.0, upper_bound - q->statistic().maxB_lower);
    }
    else {
      return r->count() * max(0.0, upper_bound - q->statistic().maxA_lower);
    }
  }

  double LowerBound(const AffinityTree *q, const AffinityTree *r) const
  {
    double dist = sqrt(q->bound().MaxDistanceSqToBound(r->bound()));
    if (unlikely(dist == 0)) {
      return DBL_MAX;
    }

    double lower_bound = 1.0 / dist;
    index_t q_end = q->first() + q->count();
    index_t r_end = r->first() + r->count();

    if (q->first() < r_end && r->first() < q_end && pref < lower_bound) {
      lower_bound = pref;
    }

    if (update_B) {
      return r->count() * max(0.0, lower_bound - q->statistic().maxB_upper);
    }
    else {
      return r->count() * max(0.0, lower_bound - q->statistic().maxA_upper);
    }
  }

  double Exact(index_t q_i, index_t r_i) const
  {
    double m;
    ArrayList<AffinityStat::Max> &maxes_temp = *maxes;

    if (unlikely(maxes_temp[r_i].index == q_i)) {
      m = maxes_temp[r_i].second;
    }
    else {
      m = maxes_temp[r_i].first;
    }

    if (unlikely(q_i == r_i)) {
      return pref - m;
    }
    else {
      DVector q_vec, r_vec;
      d_matrix.MakeColumnVector(q_i, &q_vec);
      d_matrix.MakeColumnVector(r_i, &r_vec);

      double dist = sqrt(linear::DistanceSqEuclidean(q_vec,r_vec));
      if (unlikely(dist == 0)) {
	return DBL_MAX;
      }

      return max(0.0, 1.0 / dist - m);
    }
  }
};

struct AffinitySumNodePair {
  AffinityTree *q;
  AffinityTree *r;

  double upper_bound;
  double lower_bound;

  AffinitySumNodePair *next;

  AffinitySumNodePair() {}
  ~AffinitySumNodePair() {}

  AffinitySumNodePair(AffinityTree *q_in, AffinityTree *r_in, AffinitySumProblem *problem) {
    q = q_in;
    r = r_in;
    upper_bound = problem->UpperBound(q, r);
    lower_bound = problem->LowerBound(q, r);
    next = NULL;
  }
};

AffinitySumNodePair *AffinitySumProblem::Pop()
{
  AffinitySumNodePair *p = head;
  if (likely(head != NULL)) {
    head = head->next;
    if (unlikely(!head)) {
      tail = NULL;
    }
  }
  return p;
}

void AffinitySumProblem::Push(AffinitySumNodePair *p)
{
  if (likely(tail != NULL)) {
    tail->next = p;
    tail = p;
  }
  else {
    head = tail = p;
  }
}

void affinity_sum_update(AffinityTree *q)
{
  q->statistic().sum_upper =
    max(q->left()->statistic().sum_upper, q->right()->statistic().sum_upper);
  q->statistic().sum_lower =
    min(q->left()->statistic().sum_lower, q->right()->statistic().sum_lower);
}

void affinity_sum_pass(AffinityTree *q, double upper, double lower)
{
  if (q->is_leaf()) {
    q->statistic().sum_upper += upper;
    q->statistic().sum_lower += lower;
  }
  else {
    affinity_sum_pass(q->left(), upper, lower);
    affinity_sum_pass(q->right(), upper, lower);
    affinity_sum_update(q);
  }
}

void affinity_sum_prune(AffinityTree *q, int pruned, double spent)
{
  q->statistic().remaining -= pruned;
  q->statistic().spent += spent;

  if (!q->is_leaf()) {
    affinity_sum_prune(q->left(), pruned, spent);
    affinity_sum_prune(q->right(), pruned, spent);
  }
}

void affinity_sum_base(AffinityTree *q, AffinityTree *r, AffinitySumProblem *problem) {
  index_t q_end = q->first() + q->count();
  index_t r_end = r->first() + r->count();

  ArrayList<AffinityStat::Sum> &sums = *(problem->sums);
  double sum_lower = DBL_MAX;
  double sum_upper = -DBL_MAX;

  for (index_t q_i = q->first(); q_i < q_end; q_i++) {
    double sum = sums[q_i].val;

    for (index_t r_i = r->first(); r_i < r_end; r_i++) {
      sum += problem->Exact(q_i, r_i);
    }

    sums[q_i].val = sum;

    if (sum < sum_lower) {
      sum_lower = sum;
    }
    if (sum > sum_upper) {
      sum_upper = sum;
    }
  }

  q->statistic().sum_upper += sum_upper - q->statistic().sum_upper_ex;
  q->statistic().sum_lower += sum_lower - q->statistic().sum_lower_ex;
  q->statistic().sum_upper_ex = sum_upper;
  q->statistic().sum_lower_ex = sum_lower;
}

void affinity_sum_body(AffinitySumProblem *problem)
{
  AffinitySumNodePair *p;

  while(p = problem->Pop()) {
    AffinityTree *q = p->q, *r = p->r;

    if ((p->upper_bound - p->lower_bound) / 2
	<= r->count() / q->statistic().remaining
	   * (problem->epsilon * q->statistic().sum_lower
	      - q->statistic().spent)) {
      affinity_sum_prune(q, r->count(), (p->upper_bound - p->lower_bound) / 2);
      problem->pruned_pairs++;
      problem->pruned_area += q->count() * r->count();
    }
    else {
      if (q->is_leaf() && r->is_leaf()) {
	q->statistic().sum_upper -= p->upper_bound;
	q->statistic().sum_lower -= p->lower_bound;

	affinity_sum_base(q, r, problem);
      }
      else if (q->is_leaf()) {
	AffinitySumNodePair *pA, *pB;

	pA = new AffinitySumNodePair(q, r->left(), problem);
	pB = new AffinitySumNodePair(q, r->right(), problem);

	q->statistic().sum_upper +=
	  pA->upper_bound + pB->upper_bound - p->upper_bound;
	q->statistic().sum_upper +=
	  pA->lower_bound + pB->lower_bound - p->lower_bound;

	problem->Push(pA);
	problem->Push(pB);
      }
      else if (r->is_leaf()) {
	AffinitySumNodePair *pA, *pB;

	pA = new AffinitySumNodePair(q->left(), r, problem);
	pB = new AffinitySumNodePair(q->right(), r, problem);

	affinity_sum_pass(q->left(), pA->upper_bound - p->upper_bound,
			  pA->lower_bound - p->lower_bound);
	affinity_sum_pass(q->right(), pB->upper_bound - p->upper_bound,
			  pB->lower_bound - p->lower_bound);
	affinity_sum_update(q);

	problem->Push(pA);
	problem->Push(pB);
      }
      else {
	{
	  AffinitySumNodePair *pA, *pB;

	  pA = new AffinitySumNodePair(q->left(), r->left(), problem);
	  pB = new AffinitySumNodePair(q->left(), r->right(), problem);

	  affinity_sum_pass(q->left(),
	    pA->upper_bound + pB->upper_bound - p->upper_bound,
	    pA->lower_bound + pB->lower_bound - p->lower_bound);

	  problem->Push(pA);
	  problem->Push(pB);
	}

	{
	  AffinitySumNodePair *pA, *pB;

	  pA = new AffinitySumNodePair(q->right(), r->left(), problem);
	  pB = new AffinitySumNodePair(q->right(), r->right(), problem);

	  affinity_sum_pass(q->right(),
	    pA->upper_bound + pB->upper_bound - p->upper_bound,
	    pA->lower_bound + pB->lower_bound - p->lower_bound);

	  problem->Push(pA);
	  problem->Push(pB);
	}

	affinity_sum_update(q);
      }
    }

    delete p;
  }
}

void affinity_sum_finalize(AffinityTree *d, AffinitySumProblem *problem)
{
  if (d->is_leaf()) {
    index_t d_end = d->first() + d->count();
    ArrayList<AffinityStat::Sum> &sums = *(problem->sums);
    double delta =
      ((d->statistic().sum_upper - d->statistic().sum_upper_ex)
       - (d->statistic().sum_lower - d->statistic().sum_lower_ex)) / 2;

    for (index_t d_i = d->first(); d_i < d_end; d_i++) {
      sums[d_i].val += delta;
    }
  }
  else {
    affinity_sum_finalize(d->left(), problem);
    affinity_sum_finalize(d->right(), problem);
    affinity_sum_update(d);
  }
}


void affinity_sum(double pref, const DMatrix& d_matrix, AffinityTree *d,
		  ArrayList<AffinityStat::Max> *maxesA,
		  ArrayList<AffinityStat::Max> *maxesB,
		  ArrayList<AffinityStat::Sum> *sums, bool update_B)
{
  AffinitySumProblem problem;

  if (update_B) {
    problem.Init(pref, d_matrix, d, maxesB, sums, update_B, 1e-6);
  }
  else {
    problem.Init(pref, d_matrix, d, maxesA, sums, update_B, 1e-6);
  }

  AffinitySumNodePair *p = new AffinitySumNodePair(d, d, &problem);
  affinity_sum_pass(d, p->upper_bound, p->lower_bound);
  problem.Push(p);
  affinity_sum_body(&problem);
  affinity_sum_finalize(d, &problem);

  problem.Report();
}

int main(int argc, char *argv[])
{
  double pref;
  const char *d_fname;
  int leaflen;
  DMatrix d_matrix;
  AffinityTree *d;
  bool do_naive;

  xrun_init(argc, argv);

  // PARSE INPUTS
  d_fname = xrun_param_str("d_fname");
  xrun_param_default("leaflen", "20");
  leaflen = xrun_param_int("leaflen");
  pref = xrun_param_double("pref");
  do_naive = xrun_param_exists("do_naive");

  // READING DATA
  xrun_timer_start("read_d");
  ASSERT_PASS(ReadMatrixFromText(d_fname, &d_matrix));
  xrun_timer_stop("read_d");

  // BUILDING TREE; Note: rearranges matrix elements.
  xrun_timer_start("tree_d");
  d = MakeKdTreeMidpoint<AffinityTree>(d_matrix, leaflen);
  xrun_timer_stop("tree_d");

  ArrayList<AffinityStat::Max> maxesA;
  ArrayList<AffinityStat::Max> maxesB;
  ArrayList<AffinityStat::Sum> sums;
  maxesA.Init(d_matrix.n_cols());
  maxesB.Init(d_matrix.n_cols());
  sums.Init(d_matrix.n_cols());

  xrun_timer_start("affinity_gnp");

  // RUN MAX_2
  max_2(pref, d_matrix, d, &maxesA, &maxesB, false);

  // RUN AFFINITY_SUM
  affinity_sum(pref, d_matrix, d, &maxesA, &maxesB, &sums, false);

#if 0
  // LOOP UNTIL CONVERGEANCE
  ArrayList<AffinityStat::Max> *p_maxes1 = &maxesB, *p_maxes2 = &maxesA;
  while (affinity_check(d_matrix, d, p_maxes1, &sums, &problem)) {
    // RUN AFFINITY_MAX and SUM
    affinity_max(pref, d_matrix, d, p_maxes1, p_maxes2, &sums, &problem);
    affinity_sum(pref, d_matrix, d, p_maxes1, &sums, &problem);

    ArrayList<AffinityStat::Max> *temp = p_maxes1;
    p_maxes1 = p_maxes2; p_maxes2 = temp;
  }
#endif

  xrun_timer_stop("affinity_gnp");

  delete d;

  // NAIVE
  if (do_naive) {
  }

  xrun_done();
}
