#ifndef GM_H
#define GM_H

#ifndef GM_NAMESPACE
#define GM_NAMESPACE gm
#endif

#ifndef BEGIN_GRAPHICAL_MODEL_NAMESPACE
#define BEGIN_GRAPHICAL_MODEL_NAMESPACE namespace gm {
#endif

#ifndef END_GRAPHICAL_MODEL_NAMESPACE
#define END_GRAPHICAL_MODEL_NAMESPACE }
#endif

// the following order of include statements is important
#include <fastlib/fastlib.h>
#include <boost/foreach.hpp>
#include "common_types.h"       // Set, Map, Vector, Dualmap, PriorityQueue, List
#include "value.h"              // the value of variable
#include "variable.h"           // Variable, FiniteVar
#include "universe.h"           // Universe consists of many variables
#include "assignment.h"         // Assignment is a map from Variable* to Value
#include "factor_template.h"    // Factor is a map from Assignment to a number type (double, Logarithm)
#include "factor_graph.h"       // Factor graph is a set of node and edges between variables and factors
#include "convergence_measure.h"// Specifiying stopping criteria
#include "inference.h"          // Inference algorithms: NaiveInference, SumProductInference, MessagePriorityInference
#include "logarithm.h"          // Logarithm is a nonnegative number (represented by its natural-base logarithm)

#endif // GM_H
