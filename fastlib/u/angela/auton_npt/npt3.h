/*
   File:        npt3.h
   Author:      Alexander Gray
   Description: 
*/

#ifndef NPT3_H
#define NPT3_H

#define REGULAR    0
#define SMALLFRIES 1
#define BIGCHEESES 2

#include "sheap.h"

typedef struct nodeset
{
  knode **kns;
  double ntuples;
} nodeset;

typedef struct sampling_nodeset
{
  knode **kns;
  double ntuples;
  double nmatches;
  double nsamples;
  char type;
} snodeset;

double shift_mean_of_sumsq(double sumsq1, double m1, double m2, double N);
nodeset *mk_nodeset(knode **kns, int n, double ntuples);
snodeset *mk_snodeset(knode **kns, int n, double ntuples, double nmatches,
					  double nsamples, char type);
void free_nodeset(nodeset *ns,int n);
void free_snodeset(snodeset *sns,int n);
//int compare_nodesets(void * x, void * y);
void enqueue_nodeset(sheap *sh,knode **kns, int n, double ntuples);
void enqueue_snodeset(sheap *sh,knode **kns, int n, double ntuples, 
					  double nmatches, double nsamples, char type);
void *get_from_sheap(sheap *sh);
bool sheap_is_empty(sheap *sh);

double fast_npt3(sheap *sh,
                 mapshape *ms,dym **xs,dym **ws,matcher *ma,
                 bool use_symmetry,bool use_permutes,
                 double thresh_ntuples,double connolly_thresh,
                 double *lobound,double *hibound, 
                 dyv *wlobound, dyv *whibound,dyv *wresult,dyv *wsum,dyv *wsumsq,
                 imat *permutation_cache, ivec **maps, int *starts);

#endif /* #ifndef NPT3_H */
