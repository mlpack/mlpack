/*
   File:        matcher.h
   Author:      Andrew W. Moore
   Created:     Wed May 17 12:25:12 EDT 2000
   Description: Matcher predicates for fast N-point computation

   Copyright 2000, the Auton Lab
*/

#ifndef MATCHER_H
#define MATCHER_H

/* A matcher data structure is used to describe exactly what kind of
   n-point correlation predicate we're using. 

   Information inside a matcher includes:

      The value of "n" (as in n-point)
      Distance metric (typically this will be a vector of length == number
                       of dimensions containing all 1.0's)
      A string representation of a summary of the kind of matcher
      Predicate parameters.

    Predicate parameters depend on which kind of n-point predicate we are
    using. Here are the four available types:

       1. scalar-threshold
       2. scalar-between
       3. compound-threshold
       4. compound-between

    Now let's go through the 4 types in more detail.
   
       1. scalar-threshold

        Represented on the command line as: matcher <number>
        Example:                            matcher 0.2

        This matcher matches an n-tuple of points if and only if all
        pairs of points (x_i,x_j) in the tuple satisfy

             Dist(x_i,x_j) <= <number>

        (Note, in the code it's implemented and represented as

             Dist(x_i,x_j)^2 = (x_i - x_j)^2 <= <number>^2

 
       2. scalar-between

        Represented on the command line as: matcher <number>,<number>
        Example:                            matcher 0.2,0.6

         WARNING: Either there must be no space between the , and the numbers,
                  or the expression should be in quotes:
 
                     This is fine:  matcher 0.2,0.6
                     This is fine:  matcher "0.2 , 0.6"
                     This is bad:   matcher 0.2 , 0.6
                     

        matcher p,q matches an n-tuple of points if and only if all
        pairs of points (x_i,x_j) in the tuple satisfy

             p <= Dist(x_i,x_j) <= q

        Note, in the code it's implemented and represented as

             p^2 <= (x_i - x_j)^2 <= q^2

       3. compound-threshold

           Represented on the command line as: matcher <filename>
           Example:                            matcher 3p.predicate
           Where 3p.predicate is an ascii file containing a matrix, e.g:

               0    0.1   0.5
               0.1  0     0.2
               0.5  0.2     0

           An n-tuple (x_1,x_2, .. x_n) matches the compound-threshold
           matrix H if and only if

             forall i in 1..n, and all j in i+1...n, Dist(x_i,x_j) <= H[i][j]

           The example above would match triangles in which the first two
           points were within distance 0.1 of each other, the first and third
           within distance 0.5 and the second and third within 0.2.

           Error Checking:
 
              The file MUST contain a matrix with n lines, and n numbers
              (space or comma separated) on each line, with the j'th element
              on the i'th line representing H[i][j]

              H must be symmetric

              H must have a zero diagonal

              All other entries must be strictly greater than zero.

                  
       4. compound-between

           Represented on the command line as: matcher <filename>,<filename>
           Example:                            matcher 3lo.txt,3hi.txt

           Where 3lo.txt and 3hi.txt could be, for example...

           3lo.txt:

               0    0.1   0.5
               0.1  0     0.2
               0.5  0.2     0

           3hi.txt:

               0    0.2   0.9
               0.2  0     0.21
               0.9  0.21     0

           An n-tuple (x_1,x_2, .. x_n) matches the compound-threshold
           matrix pair L,H if and only if

             forall i in 1..n, and all j in i+1...n, 
                 L[i][j] <= Dist(x_i,x_j) <= H[i][j]


   SYMMETRY: There is an important difference in the way that "scalar" versus
             "compound" predicates are counted.

             A scalar predicate neglects redundant permutations of points,
             thus if (a,b,c) matches a scalar 3pt predicate it will be counted
             only once (b,a,c) for example, will not be counted.

             A compound predicate does not neglect redundant parameters.
             The reason for this is that in the general case with different
             thresholds for different pairs within the tuple, then even
             if (a,b,c) matches the predicte, (b,a,c) (for example) might or
             might not.
   */

#include "hrect.h"
#include "distutils.h"
#include "dsut.h"

typedef struct matcher
{
  int n;
  bool between;   /* True => Range, False => Threshold */
  bool compound;  /* False => Scalar */
  double dsqd_lo; 
  double dsqd_hi; /* Squared Threshold/Upper bound */
  dym *compound_lo; 
  dym *compound_hi; /* i,j'th element is square H[i][j] value described abve */
  dyv *metric;      /* All 1's almost all the time */
  char *describe_string; /* null terminated string describing the matcher using
                             the above syntax */
} matcher;


/* A simple function that prints to stdout an English language
   description of what the distance metric means */
/* Note, if you are new to AUTON style coding, here are some 
   additional observations about this function...

         dyv's are DYnamic Vectors. See utils/amdyv.h

	 See www.cs.cmu.edu/~AUTON/programming.html for more info
*/
void explain_distance_metric(dyv *metric);

/* A simple function that prints to stdout an English language
   description of what the matcher means */
void explain_matcher(matcher *ma);


matcher *mk_symmetric_simple_matcher(int n,dyv *metric,double dist_hi);

matcher *mk_symmetric_between_matcher(int n,dyv *metric,
				      double dist_lo,double dist_hi);

matcher *mk_compound_simple_matcher(int n,dyv *metric,char *matchfile);

matcher *mk_compound_between_matcher(int n,dyv *metric,
				     char *matchfile_lo,char *matchfile);

/* Makes a matcher from a string_array. We must already know what
   distance metric we're using, and what is the "n" in "n-point"
   correlation. */
matcher *mk_matcher_from_string_array(int n,dyv *metric,string_array *sa);


/* Makes a matcher from a string. We must already know what
   distance metric we're using, and what is the "n" in "n-point"
   correlation. */
matcher *mk_matcher_from_string(int n,dyv *metric,char *s);

/* Makes a matcher from the command line. We must already know what
   distance metric we're using, and what is the "n" in "n-point"
   correlation. */
matcher *mk_matcher_from_args(int n,dyv *metric,int argc,char *argv[]);

void free_matcher(matcher *ma);

/* AG */
double row_metric_dsqd_proj(dym *x1,dym *x2,dyv *metric,int row1,int row2,
                            int projection,int projmethod);

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
                             int tuple_index_1,int tuple_index_2);
//bool matcher_test_point_pair(matcher *ma,dym *x1,dym *x2,int row1,int row2,
//	              		     int tuple_index_1,int tuple_index_2);

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
                                     ivec *permutes_ok);
//bool matcher_permute_test_point_pair(matcher *ma,dym *x1,dym *x2,
//					   int row1,int row2,
//					   int pt_tuple_index_1,int pt_tuple_index_2,
//					   imat *permutation_cache,
//					   ivec *permutes_ok);


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
			    int tuple_index_1,int tuple_index_2);

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
				    imat *permutation_cache);

bool matcher_is_symmetric(matcher *ma);

int matcher_n(matcher *ma);

char *matcher_describe_string(matcher *ma);

/* Make a cache of all permutations of this size.  For example, calling this
   with size 3 will yield an imat with the following values:
   0 1 2
   0 2 1
   1 0 2
   1 2 0
   2 0 1
   2 1 0
*/
imat *mk_permutation_cache(int size);

matcher *mk_matcher2_from_args(int n,dyv *metric,int argc,char *argv[]);

#define UNKNOWN -1 /* AG */
#define EXCLUDE 0
#define SUBSUME 1
#define INCONCLUSIVE 2

#define NONE 0
#define PARA 1
#define PERP 2
#define BOTH 3

#define NONE   0
#define WAKE   1
#define DALTON 2
#define FISHER 3
#define SUTH   4

#endif /* MATCHER_H */

