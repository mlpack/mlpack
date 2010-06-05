/** @author Dongryeol Lee
 *
 *  @brief A thin wrapper on the Matrix class with the tree.
 *
 *  @file table.h
 */

#ifndef CONTRIB_DONGRYEL_PROXIMITY_PROJECT_TABLE_H
#define CONTRIB_DONGRYEL_PROXIMITY_PROJECT_TABLE_H

namespace proximity_project {
template<typename TreeType>
class Table {
  private:
    TreeType *tree_;

    std::vector<int> old_to_new_;

    std::vector<int> new_to_old_;

  public:

    class TreeIterator {
      private:

    };

    Table() {
      tree_ = NULL;
    }

    ~Table() {
      if (tree_ != NULL) {
        delete tree_;
        tree_ = NULL;
      }
    }

    const TreeType *get_tree() const {
      return tree_;
    }

    TreeType *get_tree() {
      return tree_;
    }

    void IndexData() {

    }
};
};

#endif
