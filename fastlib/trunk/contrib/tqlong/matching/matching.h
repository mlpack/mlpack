#ifndef MATCHING_H
#define MATCHING_H

#include <string>
#include <fastlib/fastlib.h>

#ifndef MATCHING_NAMESPACE_BEGIN
#define MATCHING_NAMESPACE_BEGIN namespace match {
#endif

#ifndef MATCHING_NAMESPACE_END
#define MATCHING_NAMESPACE_END }
#endif

MATCHING_NAMESPACE_BEGIN;

std::string toString (const Vector& v);

MATCHING_NAMESPACE_END;

#endif // MATCHING_H
