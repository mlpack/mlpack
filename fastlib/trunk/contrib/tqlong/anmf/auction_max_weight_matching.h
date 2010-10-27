#ifndef AUCTION_MAX_WEIGHT_MATCHING_H
#define AUCTION_MAX_WEIGHT_MATCHING_H

#include "anmf.h"
#include <vector>

BEGIN_ANMF_NAMESPACE;

template <typename W>
    class AuctionMaxWeightMatching
{
public:
  /** typename W should implement n_rows(), n_cols(), get(i, j) */
  typedef W                    weight_matrix_type;

  /** Constructor */
  AuctionMaxWeightMatching(weight_matrix_type& weight, bool doMatch = false);

  /** return item that matches person leftIndex */
  int leftMatch(int leftIndex) const { return doneMatching_ ? leftMatch[leftIndex] : -1; }

  /** return person that matches item rightIndex */
  int rightMatch(int rightIndex) const { return doneMatching_ ? rightMatch[rightIndex] : -1; }

  /** check if matching is done */
  bool doneMatching() const { return doneMatching_; }

  /** the auction algorithm for max weight matching */
  void doMatch();
protected:
  weight_matrix_type&      weight_;
  std::vector<int>         leftMatch_, rightMatch_;
  bool                     doneMatching_;

  void forwardAuction();
private:
  std::vector<double> price_, ;
};

template <typename W>
    AuctionMaxWeightMatching<W>::AuctionMaxWeightMatching(weight_matrix_type &weight, bool match)
      : weight_(weight), doneMatching_(false),
        leftMatch_(weight.n_rows(), -1), rightMatch_(weight.n_cols(), -1)
{
  if (match)
    doMatch();
}

template <typename W>
    void AuctionMaxWeightMatching<W>::doMatch()
{
  forwardAuction();
}

template <typename W>
    void AuctionMaxWeightMatching<W>::forwardAuction()
{
  int n_rows = weight_.n_rows();
  int n_cols = weight_.n_cols();
  while (!doneMatching_)
  {
    clearBids();
    // for all unassigned person (rows)
    for (int i = 0; i < n_rows; i++) if (leftMatch_[i] == -1)
    {
      int j;
      double v, w;
      getBestAndSecondBest(i, j, v, w); // get the best and second best (benefit - price)
      placeBid(i, j, price_[j]+v-w+epsilon_);
    }
    // for all items, assign them to best bidder
    for (int j = 0; j < n_cols; j++) if ()
  }
}

END_ANMF_NAMESPACE;

#endif // AUCTION_MAX_WEIGHT_MATCHING_H
