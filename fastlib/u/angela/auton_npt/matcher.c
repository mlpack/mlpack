/*
   File:        matcher.c
   Author:      Andrew W. Moore
   Created:     Wed May 17 12:25:12 EDT 2000
   Description: Matcher predicates for fast N-point computation

   Copyright 2000, the Auton Lab
*/

#include "matcher.h"


/* A simple function that prints to stdout an English language
   description of what the distance metric means */
void explain_distance_metric(dyv *metric)
{
  int i;

  int dims = dyv_size(metric);
  printf("Distance metric:\n");
  printf("Dist(x,y) = square root of ...\n");
  for ( i = 0 ; i < dims ; i++ )
  {
    double mi = dyv_ref(metric,i);
    printf("%30s","");
    if ( mi == 1.0 )
      printf("(x[%d] - y[%d])^2",i,i);
    else
      printf("[(x[%d] - y[%d])/%g]^2",i,i,mi);
    if ( i == dims-1 )
      printf("\n");
    else
      printf(" +\n");
  }
  printf("...where x is a vector in %d-dimensional space and x[k]\n"
         "   (0 <= k < %d)\n is its k'th element\n",dims,dims);
}

/* A simple function that prints to stdout an English language
   description of what the matcher means */
void explain_matcher(matcher *ma)
{
  int i,j;
  printf("This is a %s matcher for %d-point correlation\n",
	 (ma->compound) ? "compound" : "simple",matcher_n(ma));
  printf("in which there are %s bounds on pairwise distances.\n",
	 (ma->between) ? "lower and upper" : "only upper");
  printf("A tuple of %d points { ",matcher_n(ma));
  for ( i = 0 ; i < ma -> n ; i++ )
    printf("x%d ",i);
  printf("} matches if and only if\n");

  if ( ma -> compound )
  {
    for ( i = 0 ; i < matcher_n(ma) ; i++ )
      for ( j = i+1 ; j < matcher_n(ma) ; j++ )
      {
	if ( ma -> between )
	  printf("%g <= ",sqrt(dym_ref(ma->compound_lo,i,j)));
        printf("Dist(x%d,x%d) <= %g\n",i,j,sqrt(dym_ref(ma->compound_hi,i,j)));
      }
    printf("...for some permutation of the x's.\n");
  }
  else
  {
    if ( ma -> between )
      printf("%g <= ",sqrt(ma->dsqd_lo));
    printf("Dist(xi,xj) <= %g for all i,j\n",sqrt(ma->dsqd_hi));
  }

  explain_distance_metric(ma->metric);

  printf("Matcher description: %s\n",matcher_describe_string(ma));
}

/* A utility to check that a matrix being used in a matcher is legal. */
void process_matcher_dym(int n,dym *compound,char *filename)
{
  char *errstart = mk_printf("Matrix from file %s used for a compound\n"
			     "matcher has a problem:\n",filename);
  int i,j;

  if ( dym_rows(compound) != n ) 
    my_error(mk_printf("%sIt has the wrong number of rows",errstart));

  if ( dym_cols(compound) != n ) 
    my_error(mk_printf("%sIt has the wrong number of cols",errstart));

  if ( dym_min(compound) < 0.0 )
    my_error(mk_printf("%sAt least one component is negative.",errstart));

  if ( !is_dym_symmetric(compound) )
    my_error(mk_printf("%sIt's not symmetric.",errstart));
    
  for ( i = 0 ; i < n ; i++ )
  {
    if ( dym_ref(compound,i,i) != 0.0 )
      my_error(mk_printf("%sThere's a non-zero diagonal element",errstart));
  }

  for ( i = 0 ; i < n ; i++ )
    for ( j = 0 ; j < n ; j++ )
      dym_set(compound,i,j,real_square(dym_ref(compound,i,j)));

  free_string(errstart);
}

matcher *mk_undefined_matcher(int n,dyv *metric)
{
  matcher *ma = AM_MALLOC(matcher);

  ma -> n = n;

  ma -> between = -77;
  ma -> compound = -77;
  ma -> dsqd_lo = -77.7;
  ma -> dsqd_hi = -77.7;
  ma -> compound_lo = NULL;
  ma -> compound_hi = NULL;

  ma -> metric = mk_copy_dyv(metric);

  ma -> describe_string = NULL;

  return ma;
}

matcher *mk_symmetric_simple_matcher(int n,dyv *metric,double dist_hi)
{
  matcher *ma = mk_undefined_matcher(n,metric);
  ma -> between = FALSE;
  ma -> compound = FALSE;
  ma -> dsqd_hi = dist_hi * dist_hi;
  ma -> describe_string = mk_printf("%g",dist_hi);
  return ma;
}

matcher *mk_symmetric_between_matcher(int n,dyv *metric,
				      double dist_lo,double dist_hi)
{
  matcher *ma = mk_undefined_matcher(n,metric);
  ma -> between = TRUE;
  ma -> compound = FALSE;
  ma -> dsqd_lo = dist_lo * dist_lo;
  ma -> dsqd_hi = dist_hi * dist_hi;
  ma -> describe_string = mk_printf("(%g,%g)",dist_lo,dist_hi);
  return ma;
}

dym *mk_matcher_dym_from_filename(int n,char *matchfile)
{
  int argc = 0;
  char **argv = NULL;
  dym *compound;

  if ( !file_exists(matchfile) )
    my_errorf("Can't open file %s for n-point matcher parameters\n",
	      matchfile);
  
  compound = mk_dym_from_ds_file(matchfile,argc,argv);

  process_matcher_dym(n,compound,matchfile);

  return compound;
}

matcher *mk_compound_simple_matcher(int n,dyv *metric,char *matchfile)
{
  matcher *ma = mk_undefined_matcher(n,metric);
  ma -> between = FALSE;
  ma -> compound = TRUE;
  ma -> compound_hi = mk_matcher_dym_from_filename(n,matchfile);
  ma -> describe_string = mk_printf("%s",matchfile);
  return ma;
}

matcher *mk_compound_between_matcher(int n,dyv *metric,
				     char *matchfile_lo,char *matchfile)
{
  matcher *ma = mk_undefined_matcher(n,metric);
  dym *temp;
  int i;

  ma -> between = TRUE;
  ma -> compound = TRUE;
  ma -> compound_lo = mk_matcher_dym_from_filename(n,matchfile_lo);
  ma -> compound_hi = mk_matcher_dym_from_filename(n,matchfile);
  ma -> describe_string = mk_printf("(%s,%s)",matchfile_lo,matchfile);

  temp = mk_dym_subtract(ma->compound_hi,ma->compound_lo);
  for ( i = 0 ; i < n ; i++ )
    dym_set(temp,i,i,1.0);

  if ( dym_min(temp) <= 0.0 )
    my_error("At least one component of the matcher spec leaves\n"
	     "no room for any legal distance at all\n");
  free_dym(temp);

  return ma;
}

/* Makes a matcher from a string_array. We must already know what
   distance metric we're using, and what is the "n" in "n-point"
   correlation. */
matcher *mk_matcher_from_string_array(int n,dyv *metric,string_array *sa)
{
  int size = string_array_size(sa);
  matcher *ma = NULL;
  char *s0 = (size <= 0) ? NULL : string_array_ref(sa,0);
  char *s1 = (size <= 1) ? NULL : string_array_ref(sa,1);

  if ( size < 1 || size > 2 )
    my_error("matcher description should have 1 or 2 components");
  else if ( is_a_number(s0) )
  {
    double d = atof(s0);
    if ( size == 1 )
      ma = mk_symmetric_simple_matcher(n,metric,d);
    else if ( !is_a_number(s1) )
      my_error("If lo matcher bound is numeric then so should hi");
    else
      ma = mk_symmetric_between_matcher(n,metric,d,atof(s1));
  }
  else if ( size == 1 )
    ma = mk_compound_simple_matcher(n,metric,s0);
  else
    ma = mk_compound_between_matcher(n,metric,s0,s1);

  if (Verbosity >= 0.5)
  {
    printf("I have just constructed the following matching predicate:\n");
    explain_matcher(ma);
  }

  return ma;
}


/* Replaces characters in s according to instructions in from and to.

   result[i] = s[i] unless s[i] is in "from".

   if for some j, s[i] == from[j], then result[i] = to[j]
*/
char *mk_string_replace(char *s,char *from,char *to)
{
  char *result = mk_copy_string(s);
  int i;
  int size = strlen(s);
  if ( strlen(from) != strlen(to) )
    my_error("mk_string_replace: from and to should be same length");

  for ( i = 0 ; i < size ; i++ )
  {
    int j = index_of_char(from,s[i]);
    if ( j >= 0 )
      result[i] = to[j];
  }

  return result;
}

/* Makes a matcher from a string. We must already know what
   distance metric we're using, and what is the "n" in "n-point"
   correlation. */
matcher *mk_matcher_from_string(int n,dyv *metric,char *s)
{
  char *s1 = mk_string_replace(s,"()[]{},","       ");
  string_array *sa = mk_broken_string(s1);
  matcher *ma = mk_matcher_from_string_array(n,metric,sa);
  free_string_array(sa);
  free_string(s1);
  return ma;
}

/* Makes a matcher from the command line. We must already know what
   distance metric we're using, and what is the "n" in "n-point"
   correlation. */
matcher *mk_matcher_from_args(int n,dyv *metric,int argc,char *argv[])
{
  char *match_string = string_from_args("matcher",argc,argv,"0.05");
  matcher *ma = mk_matcher_from_string(n,metric,match_string);
  return ma;
}

void free_matcher(matcher *ma)
{
  free_dyv(ma->metric);
  if ( ma -> compound_lo != NULL ) free_dym(ma->compound_lo);
  if ( ma -> compound_hi != NULL ) free_dym(ma->compound_hi);
  free_string(ma->describe_string);
  AM_FREE(ma,matcher);
}

/* AG */
double row_metric_dsqd_proj(dym *x1,dym *x2,dyv *metric,int row1,int row2,
                            int projection,int projmethod)
{
  int i, dim = dym_cols(x1); double result = 0.0, theta = 0.0;

  if (projmethod == WAKE) 
  {
    double x1norm = 0.0, x2norm = 0.0, prod = 0.0;

    for ( i = dim-1 ; i >= 0 ; i-- )  /* NOTE: NO METRIC */
    {
      double x1_i = dym_ref(x1,row1,i), x2_i = dym_ref(x2,row2,i);

      prod += x1_i * x2_i; x1norm += x1_i * x1_i; x2norm += x2_i * x2_i; 
    }
    Num_pt_dists += 3;
    
    x1norm = sqrt(x1norm); x2norm = sqrt(x2norm);
    
    if ((x1norm == 0.0) || (x2norm == 0.0)) theta = 0.0;
    else theta = acos( prod / ( x1norm * x2norm ) );

    if (projection == PERP) result = (x1norm + x2norm) * sin(theta/2.0); else
    if (projection == PARA) result = fabs(x1norm - x2norm); else
    my_error("bad projection option");
    
    result = result * result;
  }

  else if (projmethod == DALTON)
  {
    double x1norm = 0.0, x2norm = 0.0, dsqd = 0.0, dpara = 0.0;

    for ( i = dim-1 ; i >= 0 ; i-- )  /* NOTE: NO METRIC */
    {
      double x1_i = dym_ref(x1,row1,i), x2_i = dym_ref(x2,row2,i);
      double d = ( x1_i - x2_i );           

      dsqd += d * d; x1norm += x1_i * x1_i; x2norm += x2_i * x2_i; 
    }
    Num_pt_dists += 3;
    
    x1norm = sqrt(x1norm); x2norm = sqrt(x2norm); 
    dpara = fabs(x1norm - x2norm);
    
    if (projection == PERP) result = sqrt(dsqd - dpara*dpara); else
    if (projection == PARA) result = dpara; else
    my_error("bad projection option");
    
    result = result * result;
  }

  else if (projmethod == FISHER)
  {
    double x1norm = 0.0, x2norm = 0.0, dsqd = 0.0, prod = 0.0, dpara = 0.0;

    for ( i = dim-1 ; i >= 0 ; i-- )  /* NOTE: NO METRIC */
    {
      double x1_i = dym_ref(x1,row1,i), x2_i = dym_ref(x2,row2,i);
      double d = ( x1_i - x2_i );           

      prod += x1_i * x2_i; x1norm += x1_i * x1_i; x2norm += x2_i * x2_i; 
      dsqd += d * d; 
    }
    Num_pt_dists += 3;
    
    dpara = (x1norm + x2norm) / sqrt(x1norm + x2norm + 2*prod);

    if (projection == PERP) result = sqrt(dsqd - dpara*dpara); else
    if (projection == PARA) result = dpara; else
    my_error("bad projection option");
    
    result = result * result;
  }

  else if (projmethod == SUTH)
  {
    double dist = 0.0, mnorm = 0.0, prod = 0.0;

    for ( i = dim-1 ; i >= 0 ; i-- )  /* NOTE: NO METRIC */
    {
      double x1_i = dym_ref(x1,row1,i), x2_i = dym_ref(x2,row2,i);
      double d = ( x1_i - x2_i );           
      double m = ( x1_i + x2_i ) / 2.0;
    
      prod += d * m; dist += d * d; mnorm += m * m;
    }
    Num_pt_dists += 3;
    
    dist = sqrt(dist); mnorm = sqrt(mnorm);
    
    if ((dist == 0.0) || (mnorm == 0.0)) theta = 0.0;
    else theta = acos( prod / ( dist * mnorm ) );
    
    if (projection == PARA) result = dist * sin(theta - PI/2.0); else
    if (projection == PERP) result = dist * cos(theta - PI/2.0); else
    my_error("bad projection option");
    
    result = result * result;
  }
  else my_error("bad projection method option");

  return result;
}

/* Suppose we are in the process of finding out if some tuple
   matches our n-point predicate. Suppose the tuple_index_1'th element
   of the tuple is from the row1'th row of the dataset.
   And suppose the tuple_index_2'th element
   of the tuple is from the row2'th row of the dataset.

   This function finds out whether this particular pair of tuple members
   causes a violation. If it matches successfully, returns TRUE. If 
   violation, returns FALSE. 

   Note that for an entire tuple to match it would be necessary (and
   sufficient) for
   all (n choose 2) tuple-pairs to pass this test. 
*/
bool matcher_test_point_pair(matcher *ma,int projection,int projmethod,
                             dym *x1,dym *x2,int row1,int row2,
                             int tuple_index_1,int tuple_index_2)
                             
{
  bool matches;
  double dsqd = -1, dsqd_lo = 7e77, dsqd_hi;

  if (projection == NONE) 
    dsqd = row_metric_dsqd(x1,x2,ma->metric,row1,row2);
  else 
    dsqd = row_metric_dsqd_proj(x1,x2,ma->metric,row1,row2,
                                projection,projmethod);

  if ( ma -> compound )
  {
    dsqd_hi = dym_ref(ma->compound_hi,tuple_index_1,tuple_index_2);
    if ( ma -> between )
      dsqd_lo = dym_ref(ma->compound_lo,tuple_index_1,tuple_index_2);
  }
  else
  {
    dsqd_hi = ma -> dsqd_hi;
    if ( ma -> between )
      dsqd_lo = ma -> dsqd_lo;
  }

  matches = dsqd <= dsqd_hi;
  if ( ma->between && matches )
    matches = dsqd >= dsqd_lo;

  return matches;
}

/* A recursive function.  At each iteration, it adds all possible indices 
   into the kth element of the permutation.  When it gets to the end, it
   copies the newly found permutation into the permutation cache.
*/
void build_permutation_cache_recurse(int k, int *permute_index, ivec *permute,
				     imat *permutation_cache)
{
  int i,j;

  /* loop over possible values to put in the kth spot of the permutation */
  for (i=0;i<ivec_size(permute);i++)
  {
    bool ok = TRUE;

    /* check if i has already been used in this permutation */
    for (j=0;j<k;j++) if (ivec_ref(permute,j) == i) ok = FALSE;

    if (ok)
    {
      ivec_set(permute,k,i);
      if (k == (ivec_size(permute)-1)) /* we found a permutation */
      {
	copy_ivec_to_imat_row(permute,permutation_cache,*permute_index);
	(*permute_index)++;
      }
      else build_permutation_cache_recurse(k+1,permute_index,permute,
					   permutation_cache);
    }
  }
}

/* Make a cache of all permutations of this size.  For example, calling this
   with size 3 will yield an imat with the following values:
   0 1 2
   0 2 1
   1 0 2
   1 2 0
   2 0 1
   2 1 0
*/
imat *mk_permutation_cache(int size)
{
  int permute_index = 0;
  ivec *permute = mk_constant_ivec(size,-1);
  imat *permutation_cache;
  int n_permutes = 1;
  int i;

  for (i=size;i>1;i--) n_permutes *= i;

  permutation_cache = mk_constant_imat(n_permutes,size,-1);

  build_permutation_cache_recurse(0,&permute_index,permute,permutation_cache);

  free_ivec(permute);
  
  return permutation_cache;
}

/* Suppose we are in the process of finding out if some tuple
   matches our n-point predicate. Suppose the pt_tuple_index_1'th element
   of the tuple is from the row1'th row of the dataset.
   And suppose the pt_tuple_index_2'th element
   of the tuple is from the row2'th row of the dataset.

   This function finds out whether this particular pair of tuple members
   causes a violation. If it matches successfully, returns TRUE. If 
   violation, returns FALSE. 

   Note that for an entire tuple to match it would be necessary (and
   sufficient) for
   all (n choose 2) tuple-pairs to pass this test. 

   This function is a copy of matcher_test_point_pair except that it tests
   all permutations of the template against these two points to see which
   pass, and indicates the results in permutes_ok.  Additionally, it
   returns TRUE or FALSE based on whether any of the possible permutations
   still work.  
*/
bool matcher_permute_test_point_pair(matcher *ma,int projection,int projmethod,
                                     dym *x1,dym *x2,int row1,int row2,
                                     int pt_tuple_index_1,int pt_tuple_index_2,
                                     imat *permutation_cache,
                                     ivec *permutes_ok)
{
  int i;
  bool matches;
  double dsqd = -1, dsqd_lo = 7e77, dsqd_hi;
  bool any_matches = FALSE;

	/* Computing the actual pair distance */
  if (projection == NONE) {
    dsqd = row_metric_dsqd(x1,x2,ma->metric,row1,row2);
	}
  else {
    dsqd = row_metric_dsqd_proj(x1,x2,ma->metric,row1,row2,projection,projmethod);
	}

  /* make sure the cache is ready */
  if (permutation_cache && imat_cols(permutation_cache) != matcher_n(ma))
  {
    free_imat(permutation_cache);
    permutation_cache = NULL;
  }
  if (!permutation_cache) 
    permutation_cache = mk_permutation_cache(matcher_n(ma));

  /* check all permutations */
  for (i=0;i<ivec_size(permutes_ok);i++)
  {
		/* ANG ~> Changed to something more sane not containing 'continue' */
    //if (!ivec_ref(permutes_ok,i)) continue;
		
		if (ivec_ref(permutes_ok,i)) { /* If the i'th permutation might	match */
			/* Generate proper limits according to the matcher type. */
	    if ( ma -> compound ) {
			  int template_tuple_index_1,template_tuple_index_2;

		    /* Find out where the points are in the i'th permutation */
	 	  	template_tuple_index_1 = imat_ref(permutation_cache,i,pt_tuple_index_1);
	  	  template_tuple_index_2 = imat_ref(permutation_cache,i,pt_tuple_index_2);
		
	      dsqd_hi = dym_ref(ma->compound_hi,template_tuple_index_1,template_tuple_index_2);
	      if ( ma -> between ) {
					dsqd_lo = dym_ref(ma->compound_lo,template_tuple_index_1,template_tuple_index_2);
				}
	    }
	    else {
	      dsqd_hi = ma -> dsqd_hi;
	      if ( ma -> between ) {
					dsqd_lo = ma -> dsqd_lo;
				}
			}
		  /* Test for a match and update permutes_ok if needed */	
	    matches = (dsqd <= dsqd_hi);
  	  if (ma->between && matches) {
				matches = (dsqd >= dsqd_lo);
			}
	    if (matches) {
				any_matches = TRUE;
			}
  	  else {
				ivec_set(permutes_ok,i,FALSE);
			}
		}
  }

  return any_matches;
}

/* Suppose we are in the process of finding out about a tuple of
   hrects. 
   Suppose the tuple_index_1'th hrect is hr1.
   Suppose the tuple_index_2'th hrect is hr2.

   This function returns one of three values:

      EXCLUDE: This means that we have proved that for every possible
               tuple of points in which the tuple_index_1'th element is
               inside hr1 and the tuple_index_2'th element is inside hr2,
               this tuple-pair test will be violated (i.e. a call
               to matcher_test_point_pair would return FALSE) and thus 
               irrespective of all the other tuple-pair tests, such a tuple
               would fail to match the predicate.

      SUBSUME: This means that we have proved that for every possible
               tuple of points in which the tuple_index_1'th element is
               inside hr1 and the tuple_index_2'th element is inside hr2,
               this tuple-pair test will succeed (i.e. a call
               to matcher_test_point_pair would return TRUE) and thus 
               providing all the other tuple-pair tests also succeed, 
	       such a tuple would succeed in matching the predicate.
	       
      INCONCLUSIVE: It is possible that for some locations of points in
                    the hrect the test would return TRUE and other locations
                    would return FALSE.
*/
int matcher_test_hrect_pair(matcher *ma,hrect *hr1,hrect *hr2,
			    int tuple_index_1,int tuple_index_2)
{
  int i1 = tuple_index_1;
  int i2 = tuple_index_2;
  int result;
  double min_dsqd_between_hrs = hrect_min_metric_dsqd(ma->metric,hr1,hr2);
  double max_okay_matcher_dsqd = 
    (ma->compound) ? dym_ref(ma->compound_hi,i1,i2) : ma->dsqd_hi;

  if ( min_dsqd_between_hrs > max_okay_matcher_dsqd )
    result = EXCLUDE;
  else
  {
    double max_dsqd_between_hrs = hrect_max_metric_dsqd(ma->metric,hr1,hr2);
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

/* Suppose we are in the process of finding out about a tuple of
   hrects. 
   Suppose the tuple_index_1'th hrect is hr1.
   Suppose the tuple_index_2'th hrect is hr2.

   This function returns one of three values:

      EXCLUDE: This means that we have proved that for every possible
               tuple of points in which the tuple_index_1'th element is
               inside hr1 and the tuple_index_2'th element is inside hr2,
               this tuple-pair test will be violated (i.e. a call
               to matcher_test_point_pair would return FALSE) and thus 
               irrespective of all the other tuple-pair tests, such a tuple
               would fail to match the predicate.

      SUBSUME: This means that we have proved that for every possible
               tuple of points in which the tuple_index_1'th element is
               inside hr1 and the tuple_index_2'th element is inside hr2,
               this tuple-pair test will succeed (i.e. a call
               to matcher_test_point_pair would return TRUE) and thus 
               providing all the other tuple-pair tests also succeed, 
	       such a tuple would succeed in matching the predicate.
	       
      INCONCLUSIVE: It is possible that for some locations of points in
                    the hrect the test would return TRUE and other locations
                    would return FALSE.

   This is exactly like matcher_test_hrect_pair except that we test this
   set of tuples against all possible permutations of the template.  The
   final result is produced according to the following rules:

   The hierarchy is SUBSUME,INCONCLUSIVE,EXCLUDE.
   Every template permutation starts as SUBSUME.
   If you ever score lower than your current score, you get demoted to the
   new lower value.

   The return status is EXCLUDE if all are EXCLUDE, otherwise INCONCLUSIVE.
   (the caller should check at the end if there is a SUBSUME remaining, in
    which case the overall result is SUBSUME).
*/
int matcher_permute_test_hrect_pair(matcher *ma,hrect *hr1,hrect *hr2,
				    int pt_tuple_index_1,int pt_tuple_index_2,
				    ivec *permute_status, imat *num_incons,
				    imat *permutation_cache)
{
  int i;
  int result = EXCLUDE;
  double min_dsqd_between_hrs = hrect_min_metric_dsqd(ma->metric,hr1,hr2);
  double max_dsqd_between_hrs = hrect_max_metric_dsqd(ma->metric,hr1,hr2);

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

bool matcher_is_symmetric(matcher *ma)
{
  return !ma->compound;
}

int matcher_n(matcher *ma)
{
  return ma->n;
}

char *matcher_describe_string(matcher *ma)
{
  return ma->describe_string;
}

matcher *mk_matcher2_from_args(int n,dyv *metric,int argc,char *argv[])
{
  char *match_string = string_from_args("matcher2",argc,argv,"0.05");
  matcher *ma = mk_matcher_from_string(n,metric,match_string);
  return ma;
}
