/*
   File:        projnpt.c
   Author:      Alexander Gray
   Description: Projected n-point 
*/

/*
  there is a flag saying which of PARA or PERP to compute

  leaf case - slow_npt() is modifed:
    now it is called with projection flag, NONE, PARA, or PERP, which determines
      whether to use matcher_test_point_pair_para/perp() for each n-tuple of pts
    matcher_test_point_pair() is also modified to use this flag
    finally, row_metric_dsqd_proj() uses this flag

  internal node case - matcher_test_hrect_pair_proj():
    now computes min/max dist_para and min/max dist_perp instead of just 
      min dist and max dist between hrects - to share this cost between both para
      and perp cases, pull it out of matcher_test_hrect_pair_proj(), to create
      min_and_max_dsqd_proj()
    current method (which sucks): try min/max over all hyper-corner pairings
      to get min_dsqd_para_between_hrs and min_dsqd_perp_between_hrs (this is
      all that's needed for permute case as well)
      i believe this gives true lower/upper bounds though i'm not totally sure
      it's definitely brutish - 2^D x 2^D dot-products
      however at least a recursive procedure is possible, which is quite elegant
*/

#include "npt.h"
#include "projnpt.h"

extern bool Draw_joiners;
extern int Projection;
extern int Projmethod;
extern int Next_n;
extern bool Do_rectangle_animation;
extern mrkd *Mr_root;

double hrect_norm(dyv *metric, hrect *hr, int dist_type)
{
  int d, dim = dyv_size(metric);
  double result = 0.0;
  dyv *lo = hr->lo, *hi = hr->hi;

  for ( d = dim-1 ; d >= 0 ; d-- )  /* NOTE: NO METRIC */
  {
    double lo_d = dyv_ref(lo,d), hi_d = dyv_ref(hi,d);
    
    if ( dist_type == MIN_DIST )
      result += real_min( lo_d*lo_d, hi_d*hi_d );

    else if ( dist_type == MAX_DIST )
      result += real_max( lo_d*lo_d, hi_d*hi_d );

    else my_error("bad dist_type");
  }
  Num_hr_dists += 1;

  return sqrt(result);
}

double min_hrect_norm(dyv *metric, hrect *hr)
{
  return hrect_norm(metric,hr,MIN_DIST);
}

double max_hrect_norm(dyv *metric, hrect *hr)
{
  return hrect_norm(metric,hr,MAX_DIST);
}

double hrect_hrect_dot_product(dyv *metric, hrect *hr1, hrect *hr2,
                                     int dist_type)
{
  int d, dim = dyv_size(metric);
  double result = 0.0;
  dyv *lo1 = hr1->lo, *hi1 = hr1->hi, *lo2 = hr2->lo, *hi2 = hr2->hi;

  for ( d = dim-1 ; d >= 0 ; d-- )  /* NOTE: NO METRIC */
  {
    double lo1_d = dyv_ref(lo1,d), hi1_d = dyv_ref(hi1,d);
    double lo2_d = dyv_ref(lo2,d), hi2_d = dyv_ref(hi2,d);
    
    if ( dist_type == MIN_DIST )
      result += real_min( real_min( lo1_d*lo2_d, lo1_d*hi2_d ),
                          real_min( hi1_d*lo2_d, hi1_d*hi2_d ) );

    else if ( dist_type == MAX_DIST )
      result += real_max( real_max( lo1_d*lo2_d, lo1_d*hi2_d ),
                          real_max( hi1_d*lo2_d, hi1_d*hi2_d ) );

    else my_error("bad dist_type");
  }
  Num_hr_dists += 1;

  return result;
}

double min_hrect_hrect_dot_product(dyv *metric, hrect *hr1, hrect *hr2)
{ 
  return hrect_hrect_dot_product(metric,hr1,hr2,MIN_DIST);
}

double max_hrect_hrect_dot_product(dyv *metric, hrect *hr1, hrect *hr2)
{
  return hrect_hrect_dot_product(metric,hr1,hr2,MAX_DIST);
}

/* computes bounds on EITHER the parallel distance or the perp. distance */
void min_and_max_dsqd_proj(dyv *metric, hrect *hr1, hrect *hr2,
                           double *min_dsqd_between_hrs,
                           double *max_dsqd_between_hrs,
                           int projection, int projmethod)
{
  double min1 = min_hrect_norm(metric,hr1), max1 = max_hrect_norm(metric,hr1);
  double min2 = min_hrect_norm(metric,hr2), max2 = max_hrect_norm(metric,hr2);

  if (projmethod == WAKE)
  {
    if (projection == PARA) {
      *max_dsqd_between_hrs = real_max( max1 - min2, max2 - min1 );
      *min_dsqd_between_hrs = real_max( real_max( min1 - max2, min2 - max1 ), 
                                        0.0);
    }
    else if (projection == PERP) {
      double ratio, max_dp, max_theta, min_dp, min_theta;

      max_dp = max_hrect_hrect_dot_product(metric,hr1,hr2);
      if (max_dp >= 0.0) ratio = real_min( max_dp/(min1*min2), 1.0 );
      else ratio = real_max( max_dp/(max1*max2), -1.0 );
      min_theta = acos( ratio );

      min_dp = min_hrect_hrect_dot_product(metric,hr1,hr2);
      if (min_dp <= 0.0) ratio = real_max( min_dp/(min1*min2), -1.0 );
      else ratio = real_min( min_dp/(max1*max2), 1.0 );
      max_theta = acos( ratio );

      *max_dsqd_between_hrs = (max1 + max2) * sin( max_theta/2.0 );
      *min_dsqd_between_hrs = (min1 + min2) * sin( min_theta/2.0 );
    }
    else my_error("bad projection option");

    *max_dsqd_between_hrs = *max_dsqd_between_hrs * *max_dsqd_between_hrs;
    *min_dsqd_between_hrs = *min_dsqd_between_hrs * *min_dsqd_between_hrs;
  }

  else if (projmethod == DALTON)
  {
    if (projection == PARA) {
      *max_dsqd_between_hrs = real_max( max1 - min2, max2 - min1 );
      *min_dsqd_between_hrs = real_max( real_max( min1 - max2, min2 - max1 ), 
                                        0.0);
    }
    else if (projection == PERP) {
      double min_dsqd, max_dsqd, min_dsqd_para, max_dsqd_para;

      min_dsqd = hrect_min_metric_dsqd(metric,hr1,hr2);
      max_dsqd = hrect_max_metric_dsqd(metric,hr1,hr2);
      min_and_max_dsqd_proj(metric,hr1,hr2, &min_dsqd_para, &max_dsqd_para, 
                            PARA, DALTON);

      *max_dsqd_between_hrs = sqrt(real_max(0.0,max_dsqd - min_dsqd_para));
      *min_dsqd_between_hrs = sqrt(real_max(0.0,min_dsqd - max_dsqd_para));
    }
  }

  else if (projmethod == FISHER)
  {
    if (projection == PARA) {
      double min_dp, max_dp;
      double min1sqd = min1*min1, max1sqd = max1*max1;
      double min2sqd = min2*min2, max2sqd = max2*max2;

      max_dp = max_hrect_hrect_dot_product(metric,hr1,hr2);
      min_dp = min_hrect_hrect_dot_product(metric,hr1,hr2);

      *max_dsqd_between_hrs = real_max(0.0, (max1sqd + max2sqd) / 
                                      sqrt(min1sqd + min2sqd + 2*min_dp));
      *min_dsqd_between_hrs = real_max(0.0, (min1sqd + min2sqd) / 
                                       sqrt(max1sqd + max2sqd + 2*max_dp));
    }
    else if (projection == PERP) {
      double min_dsqd, max_dsqd, min_dsqd_para, max_dsqd_para;

      min_dsqd = hrect_min_metric_dsqd(metric,hr1,hr2);
      max_dsqd = hrect_max_metric_dsqd(metric,hr1,hr2);
      min_and_max_dsqd_proj(metric,hr1,hr2, &min_dsqd_para, &max_dsqd_para, 
                            PARA, FISHER);

      *max_dsqd_between_hrs = sqrt(real_max(0.0,max_dsqd - min_dsqd_para));
      *min_dsqd_between_hrs = sqrt(real_max(0.0,min_dsqd - max_dsqd_para));
    }
  }

  else my_error("bad projection method option");
}

/* computes bounds on BOTH the parallel distance or the perp. distance */
void min_and_max_dsqd_proj_both(dyv *metric, hrect *hr1, hrect *hr2,
                                double *min_dsqd_between_hrs_para,
                                double *max_dsqd_between_hrs_para,
                                double *min_dsqd_between_hrs_perp,
                                double *max_dsqd_between_hrs_perp,
                                int projmethod)
{
  double min1 = min_hrect_norm(metric,hr1), max1 = max_hrect_norm(metric,hr1);
  double min2 = min_hrect_norm(metric,hr2), max2 = max_hrect_norm(metric,hr2);

  if (projmethod == WAKE)
  {
    double ratio, max_dp, min_dp, max_theta, min_theta;

    /* parallel */
    *max_dsqd_between_hrs_para = real_max( max1 - min2, max2 - min1 );
    *min_dsqd_between_hrs_para = real_max( real_max( min1 - max2, min2 - max1 ), 
                                           0.0);
    *min_dsqd_between_hrs_para = real_square(*min_dsqd_between_hrs_para);
    *max_dsqd_between_hrs_para = real_square(*max_dsqd_between_hrs_para);

    /* perpendicular */
    max_dp = max_hrect_hrect_dot_product(metric,hr1,hr2);
    if (max_dp >= 0.0) ratio = real_min( max_dp/(min1*min2), 1.0 );
    else ratio = real_max( max_dp/(max1*max2), -1.0 );
    min_theta = acos( ratio );
    
    min_dp = min_hrect_hrect_dot_product(metric,hr1,hr2);
    if (min_dp <= 0.0) ratio = real_max( min_dp/(min1*min2), -1.0 );
    else ratio = real_min( min_dp/(max1*max2), 1.0 );
    max_theta = acos( ratio );
    
    *max_dsqd_between_hrs_perp = (max1 + max2) * sin( max_theta/2.0 );
    *min_dsqd_between_hrs_perp = (min1 + min2) * sin( min_theta/2.0 );

    *min_dsqd_between_hrs_perp = real_square(*min_dsqd_between_hrs_perp);
    *max_dsqd_between_hrs_perp = real_square(*max_dsqd_between_hrs_perp);
  }

  else if (projmethod == DALTON)
  {
    double min_dsqd, max_dsqd;

    /* parallel */
    *max_dsqd_between_hrs_para = real_max( max1 - min2, max2 - min1 );
    *min_dsqd_between_hrs_para = real_max( real_max( min1 - max2, min2 - max1 ), 
                                           0.0);
    *min_dsqd_between_hrs_para = real_square(*min_dsqd_between_hrs_para);
    *max_dsqd_between_hrs_para = real_square(*max_dsqd_between_hrs_para);

    /* perpendicular */
    min_dsqd = hrect_min_metric_dsqd(metric,hr1,hr2);
    max_dsqd = hrect_max_metric_dsqd(metric,hr1,hr2);
    
    *max_dsqd_between_hrs_perp = sqrt(real_max(0.0,
                                      max_dsqd - *min_dsqd_between_hrs_para));
    *min_dsqd_between_hrs_perp = sqrt(real_max(0.0, 
                                      min_dsqd - *max_dsqd_between_hrs_para));

    *min_dsqd_between_hrs_perp = real_square(*min_dsqd_between_hrs_perp);
    *max_dsqd_between_hrs_perp = real_square(*max_dsqd_between_hrs_perp);
  }

  else if (projmethod == FISHER)
  {
    double min_dsqd, max_dsqd, min_dp, max_dp;
    double min1sqd = min1*min1, max1sqd = max1*max1;
    double min2sqd = min2*min2, max2sqd = max2*max2;

    /* parallel */
    max_dp = max_hrect_hrect_dot_product(metric,hr1,hr2);
    min_dp = min_hrect_hrect_dot_product(metric,hr1,hr2);

    *max_dsqd_between_hrs_para = real_max(0.0, (max1sqd + max2sqd) / 
                                          sqrt(min1sqd + min2sqd + 2*min_dp));
    *min_dsqd_between_hrs_para = real_max(0.0, (min1sqd + min2sqd) / 
                                          sqrt(max1sqd + max2sqd + 2*max_dp));

    *min_dsqd_between_hrs_para = real_square(*min_dsqd_between_hrs_para);
    *max_dsqd_between_hrs_para = real_square(*max_dsqd_between_hrs_para);

    /* perpendicular */
    min_dsqd = hrect_min_metric_dsqd(metric,hr1,hr2);
    max_dsqd = hrect_max_metric_dsqd(metric,hr1,hr2);
    
    *max_dsqd_between_hrs_perp = sqrt(real_max(0.0,
                                      max_dsqd - *min_dsqd_between_hrs_para));
    *min_dsqd_between_hrs_perp = sqrt(real_max(0.0, 
                                      min_dsqd - *max_dsqd_between_hrs_para));

    *min_dsqd_between_hrs_perp = real_square(*min_dsqd_between_hrs_perp);
    *max_dsqd_between_hrs_perp = real_square(*max_dsqd_between_hrs_perp);
  }

  else my_error("bad projection method option");
}

/* just like matcher_test_hrect_pair(), but takes the distance bounds as
   arguments instead of computing them (this allows parallel or perp. distances
   to be used instead */
int matcher_test_hrect_pair_proj(matcher *ma,hrect *hr1,hrect *hr2,
                                 int tuple_index_1,int tuple_index_2,
                                 double min_dsqd_between_hrs,
                                 double max_dsqd_between_hrs)
{
  int i1 = tuple_index_1;
  int i2 = tuple_index_2;
  int result;
  double max_okay_matcher_dsqd = 
    (ma->compound) ? dym_ref(ma->compound_hi,i1,i2) : ma->dsqd_hi;

  if ( min_dsqd_between_hrs > max_okay_matcher_dsqd )
    result = EXCLUDE;
  else
  {
    if ( !ma->between )
    {
      if ( max_dsqd_between_hrs < max_okay_matcher_dsqd )
        result = SUBSUME;
      else
        result = INCONCLUSIVE;
    }
    else
    {
      double min_okay_matcher_dsqd = 
        (ma->compound) ? dym_ref(ma->compound_lo,i1,i2) : ma->dsqd_lo;
      if ( max_dsqd_between_hrs < min_okay_matcher_dsqd )
        result = EXCLUDE;
      else if ( min_dsqd_between_hrs > min_okay_matcher_dsqd &&
                max_dsqd_between_hrs < max_okay_matcher_dsqd )
        result = SUBSUME;
      else
        result = INCONCLUSIVE;
    }
  }

  return result;
}

/* just like matcher_permute_test_hrect_pair(), but takes the distance bounds as
   arguments instead of computing them (this allows parallel or perp. distances
   to be used instead */
int matcher_permute_test_hrect_pair_proj(matcher *ma,hrect *hr1,hrect *hr2,
                                         int pt_tuple_index_1,
                                         int pt_tuple_index_2,
                                         ivec *permute_status, imat *num_incons,
                                         imat *permutation_cache,
                                         double min_dsqd_between_hrs,
                                         double max_dsqd_between_hrs)
{
  int i;
  int result = EXCLUDE;

  for (i=0;i<imat_rows(permutation_cache);i++)
  {
    int t1,t2;
    double max_okay_matcher_dsqd;

    if (ivec_ref(permute_status,i) == EXCLUDE) continue;

    /* get the template tuple indices for this permutation */
    t1 = imat_ref(permutation_cache,i,pt_tuple_index_1);
    t2 = imat_ref(permutation_cache,i,pt_tuple_index_2);

    max_okay_matcher_dsqd = 
      (ma->compound) ? dym_ref(ma->compound_hi,t1,t2) : ma->dsqd_hi;

    if ( min_dsqd_between_hrs > max_okay_matcher_dsqd )
      ivec_set(permute_status,i,EXCLUDE);
    else
    {
      if ( !ma->between )
      {
        if ( max_dsqd_between_hrs < max_okay_matcher_dsqd )
	    {
          /* result = SUBSUME; do nothing, if you were SUBSUME, you can stay */
          /* this test is subsume, but overall we're inconclusive since we
             don't know about the other tests yet */
          result = INCONCLUSIVE;
        }
        else 
	    {
          ivec_set(permute_status,i,INCONCLUSIVE);
          imat_increment(num_incons,i,pt_tuple_index_1,1);
          imat_increment(num_incons,i,pt_tuple_index_2,1);
          result = INCONCLUSIVE;
        }
      }
      else
      {
        double min_okay_matcher_dsqd = 
          (ma->compound) ? dym_ref(ma->compound_lo,t1,t2) : ma->dsqd_lo;
        if ( max_dsqd_between_hrs < min_okay_matcher_dsqd )
          ivec_set(permute_status,i,EXCLUDE);
        else if ( min_dsqd_between_hrs > min_okay_matcher_dsqd &&
		  max_dsqd_between_hrs < max_okay_matcher_dsqd )
        {
          /* result = SUBSUME; do nothing, if you were SUBSUME, you can stay */
          /* this test is subsume, but overall we're inconclusive since we
             don't know about the other tests yet */
          result = INCONCLUSIVE;
        }
        else
        {
          ivec_set(permute_status,i,INCONCLUSIVE);
          imat_increment(num_incons,i,pt_tuple_index_1,1);
          imat_increment(num_incons,i,pt_tuple_index_2,1);
          result = INCONCLUSIVE;
        }
      }
    }
  }
  return result;
}

/* just like slow_npt_helper(), except it can use parallel or perp. distance,
   or both at once */
double slow_npt_helper_proj(mapshape *ms,dym **xs,dym **ws, 
                            matcher *ma_para, matcher *ma_perp,
                            bool use_symmetry, int projection, int projmethod,
                            int k,int *row_indexes,ivec **rowsets,dyv *wresult,
                            dyv *wsum, dyv *wsumsq)
{
  int n = matcher_n(ma_para);
  int i;
  ivec *rowset_k = rowsets[k];
  double result = 0.0;
  bool cutoff = FALSE;

  /* The following loop iterates over all choices for the k'th
     tuple-member. What's cutoff all about? Well, if the matcher
     is symmetric and if two of the
     rowsets are from the same kdnode then we only want to count
     tuples in which the index in the first rowset is strictly
     less that the index in the second rowset. This prevents
     double-counting. So IF it's a symmetric matcher and
     rowsets[k] comes from the same knode
     as any rowset[j] for j < k then cutoff is set to TRUE (and the 
     loop stops early) as soon as index i becomes greater than 
     row_indexes[j]. */
  for ( i = 0 ; !cutoff && i < ivec_size(rowset_k) ; i++ )
  {
    int row = ivec_ref(rowset_k,i);
    int j;
    bool ok = TRUE;

    /* The following loop finds out if setting the k'th tuple-member
       to be the row'th row of x causes a conflict with any of the 
       earlier-indexed tuple-members, and if so it sets ok to FALSE
       and exits the loop. Furthermore, we're also monitoring for
       cutoff, as described in the outer loop. */
    for ( j = 0 ; !cutoff && ok && j < k ; j++ )
    {
      int row_index_j = row_indexes[j];
      int row_j = ivec_ref(rowsets[j],row_index_j);
      bool k_and_j_rows_from_same_knode = 
        ((long)(rowsets[k])) == ((long)(rowsets[j]));

      cutoff = use_symmetry && 
	       k_and_j_rows_from_same_knode && row_index_j <= i;

      if (projection != BOTH)
        ok = !cutoff && matcher_test_point_pair(ma_para,projection,projmethod,
                                                xs[k],xs[j],row,row_j,k,j);
      else
        ok = !cutoff && matcher_test_point_pair(ma_para,PARA,projmethod,
                                                xs[k],xs[j],row,row_j,k,j)
                     && matcher_test_point_pair(ma_perp,PERP,projmethod,
                                                xs[k],xs[j],row,row_j,k,j);

      if ( !Do_rectangle_animation && (Verbosity >= 1.0 && ms != NULL) )
      {
        if ( ok )
          ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,
                                   (cutoff)?AG_BLUE:(ok)?AG_RED:AG_GREEN);
        ms_point_in_dym_colored(ms,xs[k],row,DOT_MARKTYPE,AG_BLACK);
        ms_point_in_dym_colored(ms,xs[j],row_j,DOT_MARKTYPE,AG_BLACK);
      }

      if ( !cutoff && ok && Draw_joiners )
        ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,AG_BLUE);
    }

    /* Now, we will do more work ONLY IF there's no constraint violation,
       and no double-counting... */
    if ( !cutoff && ok )
    {
      if ( k == n-1 )
      {
        result += 1.0; /* base case of recursion */
        if (wresult || wsum || wsumsq)
        {
          int wi,ki;
          double wsofar;
          
          row_indexes[k] = i;  /* just for the loop below */
          for (wi=0;wi<dyv_size(wresult);wi++)
	      {
            double temp;
            wsofar = 1.0;
	    
            for (ki=0;ki<n;ki++)
	        {
              temp = dym_ref(ws[ki],ivec_ref(rowsets[ki],row_indexes[ki]),wi);
              wsofar *= temp;
              if (wsum) dyv_increment(wsum,wi,temp);
              if (wsumsq) dyv_increment(wsumsq,wi,temp*temp);
            }
            dyv_increment(wresult,wi,wsofar);
          }
          row_indexes[k] = -77; /* prevents accidental uses */
        }
      }
      else
      {
        row_indexes[k] = i; /* recursive case */
        result += slow_npt_helper_proj(ms,xs,ws,ma_para,ma_perp,
                                       use_symmetry,projection,projmethod,
                                       k+1,row_indexes,rowsets,
                                       wresult,wsum,wsumsq);
        row_indexes[k] = -77; /* Just to prevent anyone 
                                 accidently using row_indexes[k] again */
      }
    }
  }

  return result;
}

/* just like slow_npt() except it calls slow_npt_helper_proj() instead of 
   slow_npt_helper() */
double slow_npt_proj(mapshape *ms,dym **xs,dym **ws,
                     matcher *ma_para,matcher *ma_perp,
                     bool use_symmetry,int projection,int projmethod,
                     ivec **rowsets,dyv *wresult, dyv *wsum,dyv *wsumsq)
{
  int rows[MAX_N];
  int i;
  double result;

  if ( matcher_n(ma_para) > MAX_N ) my_error("MAX_N too small");

  for ( i = 0 ; i < matcher_n(ma_para) ; i++ ) rows[i] = -77;

  result = slow_npt_helper_proj(ms,xs,ws,ma_para,ma_perp,use_symmetry,projection,
                                projmethod,0,rows,rowsets,wresult,wsum,wsumsq);
                           
  return result;
}
  
/* just like slow_permute_npt_helper(), except it can use parallel or perp. 
   distance, or both at once */
double slow_permute_npt_helper_proj(mapshape *ms,dym **xs,dym **ws,
                                    matcher *ma_para,matcher *ma_perp,
                                    int projection,int projmethod,int k,
                                    int *row_indexes,ivec **rowsets,
                                    imat *permutation_cache,ivec *permutes_ok,
                                    dyv *wresult,dyv *wsum,dyv *wsumsq)
{
  int n = matcher_n(ma_para);
  int i;
  ivec *rowset_k = rowsets[k];
  double result = 0.0;
  bool cutoff = FALSE;
  ivec *permutes_ok_copy = mk_copy_ivec(permutes_ok);

  /* The following loop iterates over all choices for the k'th
     tuple-member. What's cutoff all about? Well, if two of the
     rowsets are from the same kdnode then we only want to count
     tuples in which the index in the first rowset is strictly
     less that the index in the second rowset. This prevents
     double-counting. So if rowsets[k] comes from the same knode
     as any rowset[j] for j < k then cutoff is set to TRUE (and the 
     loop stops early) as soon as index i becomes greater than 
     row_indexes[j]. */
  for ( i = 0 ; !cutoff && i < ivec_size(rowset_k) ; i++ )
  {
    int row = ivec_ref(rowset_k,i);
    int j;
    bool ok = TRUE;

    /* The following loop finds out if setting the k'th tuple-member
       to be the row'th row of x causes a conflict with any of the 
       earlier-indexed tuple-members, and if so it sets ok to FALSE
       and exits the loop. Furthermore, we're also monitoring for
       cutoff, as described in the outer loop. */
    copy_ivec(permutes_ok,permutes_ok_copy);
    for ( j = 0 ; !cutoff && ok && j < k ; j++ )
    {
      int row_index_j = row_indexes[j];
      int row_j = ivec_ref(rowsets[j],row_index_j);
      bool k_and_j_rows_from_same_knode = 
        ((long)(rowsets[k])) == ((long)(rowsets[j]));

      cutoff = k_and_j_rows_from_same_knode && row_index_j <= i;

      if (projection != BOTH)
        ok = !cutoff && matcher_permute_test_point_pair(ma_para,
                                                        projection,projmethod,
                                                      xs[k],xs[j],row,row_j,k,j,
                                                      permutation_cache,
                                                      permutes_ok_copy);
      else
        ok = !cutoff && matcher_permute_test_point_pair(ma_para,PARA,projmethod,
                                                      xs[k],xs[j],row,row_j,k,j,
                                                      permutation_cache,
                                                      permutes_ok_copy)
                     && matcher_permute_test_point_pair(ma_perp,PERP,projmethod,
                                                      xs[k],xs[j],row,row_j,k,j,
                                                      permutation_cache,
                                                      permutes_ok_copy);

      if ( !Do_rectangle_animation && (Verbosity >= 1.0 && ms != NULL) )
      {
        if ( ok )
          ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,
                                   (cutoff)?AG_BLUE:(ok)?AG_RED:AG_GREEN);
        ms_point_in_dym_colored(ms,xs[k],row,DOT_MARKTYPE,AG_BLACK);
        ms_point_in_dym_colored(ms,xs[j],row_j,DOT_MARKTYPE,AG_BLACK);
      }

      if ( !cutoff && ok && Draw_joiners )
        ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,AG_BLUE);
    }

    /* Now, we will do more work ONLY IF there's no constraint violation,
       and no double-counting... */
    if ( !cutoff && ok )
    {
      if ( k == n-1 )
      {
        result += 1.0; /* base case of recursion */
	if (wresult || wsum || wsumsq)
	{
	  int wi,ki;
	  double wsofar;

	  row_indexes[k] = i;  /* just for the loop below */
	  for (wi=0;wi<dyv_size(wresult);wi++)
	  {
	    double temp;
	    wsofar = 1.0;

	    for (ki=0;ki<n;ki++)
	    {
	      temp = dym_ref(ws[ki],ivec_ref(rowsets[ki],row_indexes[ki]),wi);
	      wsofar *= temp;
	      if (wsum) dyv_increment(wsum,wi,temp);
	      if (wsumsq) dyv_increment(wsumsq,wi,temp*temp);
	    }
	    dyv_increment(wresult,wi,wsofar);
	  }
	  row_indexes[k] = -77; /* prevents accidental uses */
	}
      }
      else
      {
        row_indexes[k] = i; /* recursive case */
        result += slow_permute_npt_helper_proj(ms,xs,ws,ma_para,ma_perp,
                                               projection,projmethod,
                                          k+1,row_indexes,rowsets,
                                          permutation_cache,permutes_ok_copy,
                                          wresult,wsum,wsumsq);
        row_indexes[k] = -77; /* Just to prevent anyone 
                                 accidently using row_indexes[k] again */
      }
    }
  }
  free_ivec(permutes_ok_copy);

  return result;
}

/* just like slow_permute_npt() except it calls slow_permute_npt_helper_proj() 
   instead of slow_permute_npt_helper() */
double slow_permute_npt_proj(mapshape *ms, dym **xs,dym **ws, 
                             matcher *ma_para,matcher *ma_perp, 
                             int projection, int projmethod, ivec **rowsets,
                             imat *permutation_cache,dyv *wresult,
                             dyv *wsum, dyv *wsumsq)
{
  int rows[MAX_N];
  int i;
  double result;
  int num_permutes = 1;
  ivec *permutes_ok;

  if ( matcher_n(ma_para) > MAX_N ) my_error("MAX_N too small");

  for (i=matcher_n(ma_para); i>1; i--) num_permutes *= i;
  permutes_ok = mk_constant_ivec(num_permutes,1);

  for ( i = 0 ; i < matcher_n(ma_para) ; i++ ) rows[i] = -77;

  result = slow_permute_npt_helper_proj(ms,xs,ws,ma_para,ma_perp,
                                        projection,projmethod,
                                   0,rows,rowsets,
                                   permutation_cache,permutes_ok,wresult,
                                   wsum,wsumsq);

  free_ivec(permutes_ok);
  
  return result;
}
  
/* just like fast_npt() except mainly for the pruning loop, which defines
   exclusion and subsumption pruning differently in light of the simultaneous
   para/perp matching constraint, which consequently affects the way 
   num_subsumes is defined */
int fast_npt_proj(mapshape *ms,dym **xs,dym **ws,
                  matcher *ma_para, matcher *ma_perp,
                  bool use_symmetry,bool use_permutes,knode **kns,
                  double thresh_ntuples,double connolly_thresh,
                  double *lobound,double *hibound, dyv *wlobound, dyv *whibound,
                  dyv *wresult,dyv *wsum,dyv *wsumsq,
                  imat *permutation_cache,int depth, 
                  int projection,int projmethod)
{
  bool do_weights = (wresult && wlobound && whibound && kns[0]->sum_weights);
  int n = matcher_n(ma_perp);
  double result = -7.777e77; /* Set to a wild value so we'll detect a problem
                                if some path through the code neglects to set
                                result to a value */
  dyv *wtemp_result = NULL;
  dyv *wtemp_sum = NULL;
  dyv *wtemp_sumsq = NULL;
  int i,j;
  bool answer_is_zero = FALSE; /* Will be set to TRUE if we prove an
                                  EXCLUDE exists */

  bool all_subsume = TRUE; /* Set to TRUE until proven otherwise. This
                          will eventually hold the answer to the question:
                          "do all pairs of hrects satisfy SUBSUME status?" */
  /* Note, SUBSUME and EXCLUDE are described and defined in matcher.h */

  bool all_leaves = TRUE; /* Set to TRUE until proven otherwise. Will
                             eventually answer "are all the nodes in the 
                             tuple kdtree leaf nodes? */
                
  int num_subsumes[MAX_N]; /* Eventually, num_subsumes[i] will 
                              contain the number of other nodes that
                              kns[i] subsumes. */

  bool recursed = FALSE; /* Flag used only for deciding when to printf
                            stuff */

  double ntuples = total_num_ntuples(n,(use_symmetry || use_permutes),kns);
            /* The maximum possible increment to the count that could be
               obtained from this knode tuple. Note that if the matcher
                is symmetric and the nodes are in the wrong order,
                this will be zero; use_permutations also forces the ordering
                so the parameter sent is the OR of those */

  dyv *weighted_ntuples = (do_weights) ? 
    mk_weighted_total_ntuples(n,(use_symmetry || use_permutes),kns) : NULL;

  bool not_worth_it = (ntuples <= thresh_ntuples);

  ivec *permute_status_para = (permutation_cache) ? 
    mk_constant_ivec(imat_rows(permutation_cache),SUBSUME) : NULL;
  ivec *permute_status_perp = (permutation_cache) ? 
    mk_constant_ivec(imat_rows(permutation_cache),SUBSUME) : NULL;

  /* a clunky way to figure out which node may be best to split on next. */
  imat *num_incons_para = (permutation_cache) ? 
    mk_zero_imat(imat_rows(permutation_cache),n) : NULL;
  imat *num_incons_perp = (permutation_cache) ? 
    mk_zero_imat(imat_rows(permutation_cache),n) : NULL;

  if ( not_worth_it )
  {
    if ( Verbosity >= 1.0 )
      printf("thresh_ntuples = %g, this_ntuples = %g so I am cutting.\n",
             thresh_ntuples,ntuples);
  }
  else if ( connolly_thresh > 0.0 )
  {
    bool all_diameters_below_connolly_thresh = TRUE;
    int i;
    for ( i = 0 ; all_diameters_below_connolly_thresh && i < n ; i++ )
    {
      double diameter = hrect_diameter(kns[i]->hr);
      if ( diameter > connolly_thresh )
        all_diameters_below_connolly_thresh = FALSE;
    }

    if ( all_diameters_below_connolly_thresh )
      not_worth_it = TRUE;
  }

  /* This if block only used for reporting */
  if (Verbosity >= 0.5)
  {
    if ( Next_n < 1000000000 && Num_pt_dists + Num_hr_dists > Next_n )
    {
      double ferr = compute_errfrac(*lobound,*hibound);
      printf("%9d dists. ",Num_pt_dists + Num_hr_dists);
      printf("lo = %9.5e, hibound = %9.5e ",*lobound,*hibound);
      if ( ferr < 1e4 ) printf("(ferr %9.4f)",ferr);
      else              printf("(ferr %9g)",ferr);
      printf("\n");
      Next_n *= 2;
    }
  }

  for ( i = 0 ; !not_worth_it && i < n ; i++ ) num_subsumes[i] = 0;

  /* AWM note to self: You can't quit this loop as soon as you
     realize answer_is_zero because you are trying to accumulate the
     high bound. 

     AWM another note to self. I don't believe the above comment any
     more but I haven't had time to think it through or test it out.
  */
  for ( i = 0 ; !not_worth_it && i < n ; i++ )
  {
    knode *kni = kns[i];
    if (!knode_is_leaf(kni)) all_leaves = FALSE;

    /* Same self note */
    /* Efficiency note: Some of the node-pair-tests will get repeated in
       a recursive call. Cure that problem sometime! 
       Alex (Gray) says he tried this in his implementation but wasn't 
       impressed. */
    for ( j = i+1 ; j < n ; j++ )
    { 
      knode *knj = kns[j];

      if (projection == NONE)
      {
        int status;
        matcher *ma = ma_perp; 
        ivec *permute_status = permute_status_para;
        imat *num_incons = num_incons_para;

        if (use_symmetry || !use_permutes)
        {
          status = matcher_test_hrect_pair(ma,kni->hr,knj->hr,i,j);
          if (status == EXCLUDE) answer_is_zero = TRUE;
          else if ( status == SUBSUME ) {
            num_subsumes[i] += 1; num_subsumes[j] += 1;
          }
          else all_subsume = FALSE;
        }
        else
        {
          status = matcher_permute_test_hrect_pair(ma,kni->hr,knj->hr,i,j,
                                                   permute_status,
                                                   num_incons,
                                                   permutation_cache);
          if (status == EXCLUDE) answer_is_zero = TRUE;
        }
      } 
      else if (projection == BOTH)
      {
        double min_dsqd_betw_hrs_para = FLT_MAX, max_dsqd_betw_hrs_para = 0.0;
        double min_dsqd_betw_hrs_perp = FLT_MAX, max_dsqd_betw_hrs_perp = 0.0;
        int status_para, status_perp;

        min_and_max_dsqd_proj_both(ma_perp->metric,kni->hr,knj->hr,
                                   &min_dsqd_betw_hrs_para, 
                                   &max_dsqd_betw_hrs_para,
                                   &min_dsqd_betw_hrs_perp, 
                                   &max_dsqd_betw_hrs_perp,
                                   projmethod);
        if (use_symmetry || !use_permutes)
        {
          status_para = matcher_test_hrect_pair_proj(ma_para,kni->hr,knj->hr,i,j,
                                                     min_dsqd_betw_hrs_para,
                                                     max_dsqd_betw_hrs_para);
          status_perp = matcher_test_hrect_pair_proj(ma_perp,kni->hr,knj->hr,i,j,
                                                     min_dsqd_betw_hrs_perp,
                                                     max_dsqd_betw_hrs_perp);
          if ((status_para == EXCLUDE) || (status_perp == EXCLUDE))
              answer_is_zero = TRUE;

          else if (( status_para == SUBSUME ) && (status_perp == SUBSUME)) {
            num_subsumes[i] += 1; num_subsumes[j] += 1;
          }
          else all_subsume = FALSE;
        }
        else
        {
          status_para = matcher_permute_test_hrect_pair_proj(ma_para,
                                                        kni->hr,knj->hr,i,j,
                                                        permute_status_para,
                                                        num_incons_para,
                                                        permutation_cache,
                                                        min_dsqd_betw_hrs_para,
                                                        max_dsqd_betw_hrs_para);
          status_perp = matcher_permute_test_hrect_pair_proj(ma_perp,
                                                        kni->hr,knj->hr,i,j,
                                                        permute_status_para,
                                                        num_incons_para,
                                                        permutation_cache,
                                                        min_dsqd_betw_hrs_perp,
                                                        max_dsqd_betw_hrs_perp);
          if ((status_para == EXCLUDE) || (status_perp == EXCLUDE))
              answer_is_zero = TRUE;
        }
      }
    }
  }

  /* some extra cleanup to find the real state. first we figure out if we have
     a subsume for any of the permutations.  the next part is stranger.  we're
     just trying to wedge the results into the same logic later for choosing
     which node to split on.  if a permutation was inconclusive then every
     node who was inconclusive in that permutation gets a checkmark.  in this
     case "getting a checkmark" means setting num_subsumes to what is expected
     later in the old way of doing the checking. 
  */
  if (projection == NONE) 
  {
    ivec *permute_status = permute_status_para;
    imat *num_incons = num_incons_para;

    if (!use_symmetry && use_permutes && !answer_is_zero)
    {
      all_subsume = FALSE;
      for (i=0;i<ivec_size(permute_status);i++)
        if (ivec_ref(permute_status,i) == SUBSUME) all_subsume = TRUE;
    
      if (!all_subsume)
        for (i=0;i<ivec_size(permute_status);i++)
          if (ivec_ref(permute_status,i) == INCONCLUSIVE)
            for (j=0;j<imat_cols(num_incons);j++)
              if (imat_ref(num_incons,i,j) > 0)
                num_subsumes[j] = n-1;
    }
  } 
  else if (projection == BOTH)
  {
    if (!use_symmetry && use_permutes && !answer_is_zero)
    {
      int siz = ivec_size(permute_status_para);
      all_subsume = FALSE;
      for (i=0;i<siz;i++)
        if ((ivec_ref(permute_status_para,i) == SUBSUME) &&
            (ivec_ref(permute_status_perp,i) == SUBSUME))
          all_subsume = TRUE;
    
      if (!all_subsume)
        for (i=0;i<siz;i++)
          if ((ivec_ref(permute_status_para,i) == INCONCLUSIVE) ||
              (ivec_ref(permute_status_perp,i) == INCONCLUSIVE)) 
          {
            int siz2 = imat_cols(num_incons_para);
            for (j=0;j<siz2;j++)
              if ((imat_ref(num_incons_para,i,j) > 0) ||  
                  (imat_ref(num_incons_perp,i,j) > 0))
                num_subsumes[j] = n-1;
          }
    }
  }

  /* for testing using slow_npt() only - comment out all pruning above */
  //all_subsume = FALSE;
  //for ( i = 0 ; i < n ; i++ ) {
  //   if (!knode_is_leaf(kns[i])) all_leaves = FALSE;
  //}

  /* This if block only concerned with animations */
  if ( Do_rectangle_animation || (Verbosity >= 1.0 && ms != NULL) )
  {
    int col = (not_worth_it) ? AG_CYAN : 
              (answer_is_zero) ? AG_RED :
              (all_subsume) ? AG_GREEN : 
              (all_leaves) ? AG_BLACK : AG_BLUE;
    if ( Do_rectangle_animation )
    {
      if ( col != AG_BLUE && col != AG_CYAN )
      {
        rectangle_animate(ms,kns,n,col);
        if (Verbosity >= 1.0) wait_for_key();
      }
    }
    else
    {
      ag_on("");
      draw_matcher_key(ms,ma_para);
      ag_set_pen_color(AG_BLACK);
      if ( dym_rows(xs[0]) < 27 ) draw_lettered_mrkd_points(ms,Mr_root,xs[0]);
      else
      {
        dym *ox = other_x(xs,n);
        draw_x(ms,xs[0],100000);
        if ( ox != NULL )
        {
          ag_set_pen_color(AG_RED);
          draw_x(ms,ox,100000);
        }
      }

      ag_set_pen_color(col);
      for ( i = 0 ; i < n ; i++ )
      {
        hrect *hr = kns[i]->hr;
        char *s = mk_printf("%d",i);
        double bx;
        double by;
        double cx = hrect_lo_ref(hr,0);
        double cy = hrect_lo_ref(hr,1);
        double dx,dy;
        
        mapshape_datapoint_to_agcoords(ms,cx,cy,&dx,&dy);

        bx = range_random(real_max(10.0,dx-40.0),real_min(502.0,dx+40.0));
        by = range_random(real_max(10.0,dy-40.0),real_min(502.0,dy+40.0));
        ag_print(bx,by,s);
        ag_line(bx,by,dx,dy);
        ms_hrect(ms,hr);
        free_string(s);
      }
    }
  }

  /* Okay, now we're in a position to go ahead and prune if we can */
  if ( not_worth_it )
  {
    result = 0.0;  /* and don't bother updating wresult */
    if ( Verbosity >= 1.0 )
      printf("Number of possible matches zero or below thresh. Prune\n");
    /* Note we should NOT decrease hi bound in this case */
  }
  else if ( answer_is_zero )
  {
    result = 0.0;  /* and don't bother updating wresult */
    if ( Verbosity >= 1.0 )
    {
      int kk;
      printf("nodes: ");
      for (kk=0;kk<n;kk++) printf("%3d:%3d  ",kns[kk]->lo_index,kns[kk]->hi_index);
      printf("\n");
      printf("Impossible to match. Reduce hibound by %g\n",ntuples);
      if (weighted_ntuples)
      {
        dyv *test_wresult = mk_zero_dyv(dyv_size(wresult));
        special_weighted_symmetric_debugging_test(xs,ws,ma_para,kns,test_wresult);
        fprintf_oneline_dyv(stdout,"weighted hibound reduced by",weighted_ntuples,"\n");
        if (!dyv_equal(test_wresult,weighted_ntuples))
        {
          printf("fast_npt_proj: Dyvs not equal!!\n");
          fprintf_oneline_dyv(stdout,"test_dyv",test_wresult,"\n");
          really_wait_for_key();
        }
      }
    }
    *hibound -= ntuples;
    if (do_weights) 
    {
      dyv_subtract(whibound,weighted_ntuples,whibound);
      printf("%20.10f decrease in whibound\n",dyv_ref(weighted_ntuples,0));
    }
  }
  else if ( all_subsume )
  {
    result = ntuples;
    if ( Verbosity >= 1.0 )
      printf("All tuples match. Prune. Increases lobound by %g\n",
             ntuples);
    *lobound += ntuples;
    if (do_weights) 
    {
      dyv_plus(wresult,weighted_ntuples,wresult);
      dyv_plus(wlobound,weighted_ntuples,wlobound);
    }
    if (wsum)
    {
      dyv *tmp = mk_weighted_sum_ntuples(n, (use_symmetry || use_permutes), 
					 FALSE, kns);
      dyv_plus(wsum,tmp,wsum);
      free_dyv(tmp);
    }
    if (wsumsq)
    {
      dyv *tmp = mk_weighted_sum_ntuples(n, (use_symmetry || use_permutes), 
                                         TRUE, kns);
      dyv_plus(wsumsq,tmp,wsumsq);
      free_dyv(tmp);
    }
  }
  else
  {
    /* We're going to recurse by finding one element of kns (call it kns[i]
       and calling fast_npt twice, with kns[i] replaced by kns[i]->left and
       kns[i] -> right. But what should i be? We're going to use the 
       following (debatable) strategy:
             Choose it to be the biggest non-leaf, non-all-subsuming node
             available. "biggest" means most points. Non-leaf is because
             we can't recurse on leaves. Non-all-subsuming is because
             it would be a waste to recurse on an all-subsuming node since
             it is the non-all-subsuming nodes that need to get smaller
             if we are going to be able to do a subsume-based prune. 

	     If all nodes are leaves we'll do it the slow way

             If all non-leaf nodes are subsuming, split the largest
             non-leaf node. (Andrew had a lot of code for doing something
             cleverer in this case, involving doing the slow thing over
             the leaf nodes only and multiplying by a factor derived from
             the non-leaf-but-subsume nodes. He abandonded it because it
             was too complicated to factor in permutation-overcounting-prevent
             issues. But it would be easy to reinstitute for non-symmetric
             searches). */

    int split_index = -1;
    int split_index_num_points = -77;

    for ( i = 0 ; i < n ; i++ )
    {
      bool subsume_all = num_subsumes[i] == n-1;
      if ( !subsume_all && !knode_is_leaf(kns[i]) )
      {
        int num_points = kns[i]->num_points;
        if ( split_index < 0 || num_points > split_index_num_points )
        {
          split_index = i;
          split_index_num_points = num_points;
        }
      }
    }

    /* The only way that split_index could be undefined is if all 
       non-all-subsuming knodes were leaves */

    if ( split_index < 0 )
    {
      /* We failed to find a non-subsuming non-leaf. Now we'll be happy
         with the largest non-leaf whether it sumbsumes or not. */

      for ( i = 0 ; i < n ; i++ )
      {
        if ( !knode_is_leaf(kns[i]) )
        {
          int num_points = kns[i]->num_points;
          if ( split_index < 0 || num_points > split_index_num_points )
          {
            split_index = i;
            split_index_num_points = num_points;
          }
        }
      }
    }
      
    if ( split_index < 0 )
    {
      /* All the nodes are leaves */
      ivec *rowsets[MAX_N];

      for ( i = 0 ; i < n ; i++ )
      {
        if ( !knode_is_leaf(kns[i]) ) my_error("no way jose");
        rowsets[i] = kns[i] -> rows;
      }

      if (do_weights) wtemp_result = mk_zero_dyv(dyv_size(wresult));
      if (wsum) wtemp_sum = mk_zero_dyv(dyv_size(wsum));
      if (wsumsq) wtemp_sumsq = mk_zero_dyv(dyv_size(wsumsq));

      if (use_symmetry || !use_permutes)	
        result = slow_npt_proj(ms,xs,ws,ma_para,ma_perp,
                               use_symmetry,projection,projmethod,rowsets,
                               wtemp_result,wtemp_sum,wtemp_sumsq);
      else
        result=slow_permute_npt_proj(ms,xs,ws,ma_para,ma_perp,
                                     projection,projmethod,rowsets,
                                     permutation_cache,
                                     wtemp_result,wtemp_sum,wtemp_sumsq);

      *lobound += result;
      *hibound -= (ntuples - result);

      if (do_weights)
      {
        dyv_plus(wlobound,wtemp_result,wlobound);
        dyv_subtract(whibound,weighted_ntuples,whibound);
        dyv_plus(whibound,wtemp_result,whibound);
        dyv_plus(wresult,wtemp_result,wresult);
	
        if (Verbosity >= 1.0)
        {
          dyv *test_wresult = mk_zero_dyv(dyv_size(wresult));
          special_weighted_symmetric_debugging_test(xs,ws,ma_para,kns,
                                                    test_wresult);
          fprintf_oneline_dyv(stdout,"found",wtemp_result,"\n");
          fprintf_oneline_dyv(stdout,"reduce hibound by (minus found)",
                              weighted_ntuples,"\n");
          if (!dyv_equal(test_wresult,weighted_ntuples))
          {
              printf("fast_npt_proj: Dyvs not equal!!\n");
              fprintf_oneline_dyv(stdout,"test_dyv",test_wresult,"\n");
              really_wait_for_key();
          }
        }
        free_dyv(wtemp_result);
      }
      if (wsum)
      {
        dyv_plus(wsum,wtemp_sum,wsum);
        free_dyv(wtemp_sum);
      }
      if (wsumsq)
      {
        dyv_plus(wsumsq,wtemp_sumsq,wsumsq);
        free_dyv(wtemp_sumsq);
      }
    }
    else /* There's someone waiting to be splitted... */
    {
      knode *parent = kns[split_index];
      knode *child1 = parent->left;
      knode *child2 = parent->right;

      recursed = TRUE;
      if ( !Do_rectangle_animation && Verbosity >= 1.0 )
      {
        printf("About to recurse. lobound=%g, hibound=%g\n",*lobound,*hibound);
        wait_for_key();
      }

      result = 0.0;
      kns[split_index] = child1;

      result += fast_npt_proj(ms,xs,ws,ma_para,ma_perp,
                              use_symmetry,use_permutes,kns,
                              thresh_ntuples,connolly_thresh,
                              lobound,hibound,wlobound,whibound,
                              wresult,wsum,wsumsq,permutation_cache,depth+1,
                              projection,projmethod);

      kns[split_index] = child2;

      result += fast_npt_proj(ms,xs,ws,ma_para,ma_perp,
                              use_symmetry,use_permutes,kns,
                              thresh_ntuples,connolly_thresh,
                              lobound,hibound,wlobound,whibound,
                              wresult,wsum,wsumsq,permutation_cache,depth+1,
                              projection,projmethod);
      
      kns[split_index] = parent;
    }
  }

  if ( !recursed && Verbosity >= 1.0 )
  {
    printf("lobound = %g, hibound = %g\n",*lobound,*hibound);
    wait_for_key();
  }

  if (weighted_ntuples) free_dyv(weighted_ntuples);
  if (permute_status_para) free_ivec(permute_status_para);
  if (permute_status_perp) free_ivec(permute_status_perp);
  if (num_incons_para) free_imat(num_incons_para);
  if (num_incons_perp) free_imat(num_incons_perp);

  return result;
}

