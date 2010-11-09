/** @file: anmf.h
 *  The main header file for affine NMF
 *  including following functionalities:
 *    + Graph Matching
 *    + Nonegative Matrix Factorization
 *    + etc.
 */

#include <string>

#ifndef TQLONG_ANMF_H
#define TQLONG_ANMF_H

#ifndef BEGIN_ANMF_NAMESPACE
#define BEGIN_ANMF_NAMESPACE namespace anmf {
#endif
#ifndef END_ANMF_NAMESPACE
#define END_ANMF_NAMESPACE }
#endif

BEGIN_ANMF_NAMESPACE;

std::string toString(const Vector& v);

END_ANMF_NAMESPACE;

//#include "max_weight_matching.h"
//#include "naive_distance_matrix.h"
//#include "allnn_kdtree_distance_matrix.h"
//#include "allnn_auction_max_weight_matching.h"

#endif
