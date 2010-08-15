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
#include "common_types.h"
#include "value.h"
#include "variable.h"
#include "universe.h"
#include "assignment.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;
/**
  * A domain is a list of variables, e.g. x0, x1, ... xn,
  * which could be the arguments of a function or factor
  */
typedef Vector<const Variable*> Domain;
END_GRAPHICAL_MODEL_NAMESPACE;

#include "factor_template.h"
#include "factor_graph.h"
#include "inference.h"
#include "logarithm.h"

#endif // GM_H
