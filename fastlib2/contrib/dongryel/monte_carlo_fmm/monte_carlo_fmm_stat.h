#ifndef INSIDE_MONTE_CARLO_FMM_H
#error "This is not a public header file!"
#endif

/** @brief The type of our query tree.
 */
typedef GeneralBinarySpaceTree<DBallBound < LMetric<2>, Vector>, Matrix > QueryTree;

/** @brief The type of our reference tree.
 */
typedef GeneralBinarySpaceTree<DBallBound < LMetric<2>, Vector>, Matrix > ReferenceTree;
