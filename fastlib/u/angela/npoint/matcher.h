/** 
 * @author Angela N. Grigoroaia
 * @file: matcher.h
 * @description: Matcher code for n-point correlation computations.
**/

/**
 * Main requirements: 
 *	- representation for a generic matcher 
 *	- check if a given n-tuple verifies a given matcher
 *	- NEW: structure to return detailed results from the matcher test
**/
   
/**
 * OLD AUTON EXPLANATION
 *
 * Various types of matchers are:
 *  1. Scalar upper threshold
 *	~> This matcher matches an n-tuple of points if and only if all pairs of 
 *	points (x_i,x_j) in the tuple satisfy 
 *			d(x_i,x_j) <= <number>
 *  2. Scalar between
 *	~> This matches an n-tuple of points if and only if all pairs of points 
 *	(x_i,x_j) in the tuple satisfy 
 *			p <= d(x_i,x_j) <= q
 *  3. Compound upper threshold
 *	~> An n-tuple (x_1,x_2, .. x_n) matches the compound threshold matrix H if 
 *	and only if for all i in 1..n, and all j in i+1...n, we have
 *			d(x_i,x_j) <= H[i][j]
 *	The example above would match triangles in which the first two points were 
 *	within distance 0.1 of each other, the first and third within distance 0.5 
 *	and the second and third within 0.2.
 *	~> Error Checking: The file MUST contain a matrix with n lines, and n 
 *	numbers (space or comma separated) on each line, with the j'th element on 
 *	the i'th line representing H[i,j]. The following rules apply to H
 *		- H[i,j] = H[j,i] (this is not generally true)
 *		- H[i,i] = 0 for any i
 *		- H[i,j] > 0 for any i<>j
 *	~> Represented on the command line as: --matcher=<filename>
 *	Example: --matcher=3p.predicate  where 3p.predicate is an ascii file 
 *	containing a matrix, e.g:
 *		0    0.1   0.5
 *		0.1  0     0.2
 *		0.5  0.2     0
 *  4. Compound between
 *	~> An n-tuple (x_1,x_2, .. x_n) matches the compound threshold matrix pair 
 *	L,H if and only if for all i in 1..n, and all j in i+1...n, we have 
 *			L[i][j] <= d(x_i,x_j) <= H[i][j]
 *	~> Represented on the command line as: --matcher=<filename>
 *	Example: --matcher=3p_matcher.txt 
 *
 * SYMMETRY: There is an important difference in the way that "scalar" versus 
 * "compound" predicates are counted.
 * 		A scalar predicate neglects redundant permutations of points, thus if 
 * 	(a,b,c) matches a scalar 3pt predicate it will be counted only once (b,a,c)
 * 	for example, will not be counted.
 *  	A compound predicate does not neglect redundant parameters. The reason for
 *  this is that in the general case with different thresholds for different 
 *  pairs within the tuple, then even if (a,b,c) matches the predicate, (b,a,c)
 *  (for example) might not.
**/

/**
 * Current implementation notes:
 *
 * We will store all matchers as compound between matchers because all other 
 * types can be written in this form. For example, a simple matcher for 3 point
 * correlation (i.e. n = 2) is equivalent to the compound matcher:
 *			0 2 2
 *			2 0 2
 *			2 2 0
 * Also, any matcher can be written as a between type matcher, eventually 
 * setting the lower bound to 0. Thus, the upper compound matcher:
 *			0 1 2
 *			1 0 2
 *			1 2 0
 * can be written as a compound between matcher:
 *        0 0 0						0 1 2
 *  lo =  0 0 0		   hi =	1 0 2
 *        0 0 0						1 2 0
 * 
 * For now, we will also store additional information that identifies the 
 * matcher as one of the 4 types described above. At this point, this has 
 * little use, except for making sanity chacks easier. However, it might 
 * be important for multi-bandwidth n-point.
**/


#ifndef MATCHER_H
#define MATCHER_H

#include "fastlib/fastlib.h"
#include "globals.h"
#include "metrics.h"

class Matcher {
	public:
	  int n;		/* This is the n in 'n-point' */
	  int simple;	/* This is 1 is we have a simple matcher */
	  Matrix lo; 	/* Compound lower bound */
	  Matrix hi; 	/* Compound upper bound */

	public:
	  Matcher() {} /* default constructor */
	  ~Matcher() {} /* default destructor */

  /** 
   * The most basic matcher maker that creates a simple 0-upper 
   * bound matcher of size n. 
   * NOTE: We always need to know the size of the matcher before 
   * we create it.
   */
	public:
  void Init(const int size) {
		n = size;
		simple = 1;
		lo.Init(n,n);
		lo.SetZero();
		hi.Init(n,n);
		hi.SetZero();
 	}

	/**
	 * The most general matcher maker. It needs one input file containing both
	 * bounds in the matcher. If a problem occurs it returns an empty matcher and
	 * SUCCESS_WARN. Otherwise, it returns SUCCESS_PASS. 
	 *
	 * Notes: 
	 * 		- *Both* the lo and hi bound must be given in the file in matrix form.
	 * 		- In case of a simple matcher, the file should contain only two,
	 * 		ordered, comma-separated values.
	 */
	public:
	success_t InitFromFile(const int size, const char *file);

  /**
	 * Check if 'matcher' is a valid matcher. This implies that:
	 *	- both 'lo' and 'hi' have the same dimension and that dimension should be
	 *		equal to n
	 *	- 'lo' and 'hi' should only have non-negative values
	 *	- lo[i,i] = hi[i,i] = 0 for all possible i
	 *	- if the matcher is simple the values that are not on the	main diagonal 
	 *		should be equal.
	 *	- if the matcher is an upper limit only 'lo' should be 0
	 *	- if the matcher is a between type lo[i,j] <= hi[i,j] for all possible i,j
	**/
	private:
  success_t IsValid() const;
	
  /**
   * Checks if the n-tuple composed of the colums of 'X' specified by 'index' 
	 * matches the matcher when using the given distance metric. The following
	 * situations can occur:
	 * 	- SUCCES_PASS => the matcher is satisfied.
	 * 	- SUCCESS_FAIL => the lo or hi bound check failed
  **/
	public:
	success_t Matches(const Matrix X, const Vector index, Metric metric) const;

	/**
	 * Checks if an n-tuple matches the matcher without having to specify any of
	 * the points. We only need to specify the distances (or pseudodistances) in a
	 * matrix.
	 *
	 * Note:
	 * To be used for knodes when the max and min distances between different
	 * knodes are known.
	 */
	public:
	success_t Matches(const Matrix distances) const;
};

#endif


