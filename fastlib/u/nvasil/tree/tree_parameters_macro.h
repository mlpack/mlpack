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

#define TREE_PARAMETERS(IDENTIFIER, \
		                    PRECISION,\
		                    ALLOCATOR,\
		                    METRIC,\
	                      BOUNDINGBOX,\
	                      NODESTATISTICS,\
                        POINTIDDISCRIMINATOR,\
		                    PIVOTER, \
		                    diagnostic) \
struct BasicParameters##IDENTIFIER##_t { \
 public: \
	typedef PRECISION Precision_t; \
	typedef ALLOCATOR              Allocator_t; \
	typedef METRIC<Precision_t>    Metric_t;  \
};\
\
struct NodeParameters##IDENTIFIER##_t {\
 public: \
 	typedef BOUNDINGBOX<BasicParameters##IDENTIFIER##_t, diagnostic> BoundingBox_t; \
	typedef NODESTATISTICS<BasicParameters##IDENTIFIER##_t> NodeCachedStatistics_t; \
	typedef POINTIDDISCRIMINATOR PointIdDiscriminator_t; \
}; \
\
struct ExtraTreeParameters##IDENTIFIER##_t { \
 public:	\
	typedef PIVOTER<BasicParameters##IDENTIFIER##_t, diagnostic> Pivot_t;\
};\
\
struct TreeParameters##IDENTIFIER##_t { \
 public: \
	typedef PRECISION Precision_t; \
	typedef ALLOCATOR              Allocator_t; \
	typedef METRIC<Precision_t>    Metric_t;  \
 	typedef BOUNDINGBOX<BasicParameters##IDENTIFIER##_t, diagnostic> BoundingBox_t; \
	typedef NODESTATISTICS<BasicParameters##IDENTIFIER##_t> NodeCachedStatistics_t; \
	typedef POINTIDDISCRIMINATOR PointIdDiscriminator_t; \
	typedef PIVOTER<BasicParameters##IDENTIFIER##_t, diagnostic> Pivot_t;\
};
