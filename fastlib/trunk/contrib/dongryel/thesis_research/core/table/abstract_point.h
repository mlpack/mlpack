/** @file abstract_point.h
 *
 *  A prototype for an abstract point.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_ABSTRACT_POINT_H
#define CORE_TABLE_ABSTRACT_POINT_H

namespace core {
namespace table {
class AbstractPoint {
  public:
    virtual ~AbstractPoint() {
    }

    virtual int length() const = 0;

    virtual double operator[](int i) const = 0;

    virtual void Print() const = 0;

    virtual void Init(int length) = 0;
};
};
};

#endif
