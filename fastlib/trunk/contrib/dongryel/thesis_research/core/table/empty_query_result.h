/** @file empty_query_result.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_EMPTY_QUERY_RESULT_H
#define CORE_TABLE_EMPTY_QUERY_RESULT_H

#include <boost/serialization/serialization.hpp>

namespace core {
namespace table {
class EmptyQueryResult {

  public:

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
    }

    /** @brief Aliases a subset of the given result.
     */
    template<typename TreeIteratorType>
    void Alias(TreeIteratorType &it) {
    }

    /** @brief Aliases a subset of the given result.
     */
    template<typename TreeIteratorType>
    void Alias(const EmptyQueryResult &result_in, TreeIteratorType &it) {
    }

    /** @brief Aliases another result.
     */
    void Alias(const EmptyQueryResult &result_in) {
    }

    /** @brief Copies the given result back onto the result.
     */
    void Copy(const EmptyQueryResult & result_in) {
    }
};
}
}

#endif
