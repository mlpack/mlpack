/*
 * =====================================================================================
 * 
 *       Filename:  tree_parameters_macro.h
 * 
 *    Description
 * 
 *        Version:  1.0
 *        Created:  05/03/2007 01:03:57 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#define TREE_PARAMETERS(PRECISION,\
		                    ALLOCATOR,\
		                    METRIC,\
	                      BOUNDINGBOX,\
	                      NODESTATISTICS,\
                        POINTIDDESCRIMINATOR,\
		                    PIVOTER, \
		                    diagnostic) \
struct BasicParameters { \
	typedef PRECISION Precision_t; \
	typedef ALLOCATOR              Allocator_t; \
	typedef METRIC<Precision_t>    Metric_t;  \
};\
\
struct NodeParameters {\
  typedef BOUNDINGBOX<BasicParameters, diagnostic> BoundingBox_t; \
	typedef NODESTATISTICS<BasicParameters> NodeCachedStatistics_t; \
	typedef POINTIDDISCIMINATOR PointIdDescriminator_t; \
}; \
\
struct ExtraTreeParameters { \
	typedef PIVOTER<BasicParameters> Pivot_t;\
};\
\
struct Parameters :  public BasicParamters, public NodeParameters,\
                     public ExtraTreeParameters { \
 \
};
