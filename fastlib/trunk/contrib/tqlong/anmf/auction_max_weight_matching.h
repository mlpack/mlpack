#ifndef AUCTION_MAX_WEIGHT_MATCHING_H
#define AUCTION_MAX_WEIGHT_MATCHING_H

#include "anmf.h"
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>

BEGIN_ANMF_NAMESPACE;

template <typename W>
    class AuctionMaxWeightMatching
{
public:
  /** typename W should implement n_rows(), n_cols(), get(i, j), setPrice(j, p) */
  typedef W                weight_matrix_type;

  /** Constructor */
  AuctionMaxWeightMatching(weight_matrix_type& weight, bool doMatch = false);

  /** return item that matches person leftIndex */
  int leftMatch(int leftIndex) const { return doneMatching_ ? leftMatch_[leftIndex] : -1; }

  /** return person that matches item rightIndex */
  int rightMatch(int rightIndex) const { return doneMatching_ ? rightMatch_[rightIndex] : -1; }

  /** check if matched */
  bool matched(int index) const { return index != -1; }
  void unmatch(int& index) const { index = -1; }

  /** check if matching is done */
  bool doneMatching() const { return doneMatching_; }

  /** the auction algorithm for max weight matching */
  void doMatch();
protected:
  weight_matrix_type&      weight_;
  double                   epsilon_;
  int                      n_rows_, n_cols_;
  std::vector<int>         leftMatch_, rightMatch_;
  bool                     doneMatching_;

  std::vector<double>      price_, bid_;
  std::vector<int>         winner_;

  void forwardAuction();
  void placeBid(int bidder, int item, double price);
  void clearBids();
  void getBestAndSecondBest(int bidder, int& best_item, double& best_surplus, double& second_surplus);
  void setMatch(int new_bidder, int item);
  bool checkEpsilonComplementary() const;
private:
};

template <typename W>
    AuctionMaxWeightMatching<W>::AuctionMaxWeightMatching(weight_matrix_type &weight, bool match)
      : weight_(weight),
        n_rows_(weight_.n_rows()),
        n_cols_(weight.n_cols()),
        leftMatch_(n_rows_, -1),
        rightMatch_(n_cols_, -1),
        doneMatching_(false),
        price_(n_cols_, 0),
        bid_(n_cols_, -std::numeric_limits<double>::infinity()),
        winner_(n_cols_, -1)
{
  epsilon_ = 1.0 / n_rows_;
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
  while (!doneMatching_)
  {
//    std::cout << (checkEpsilonComplementary() ? "e-CS satisfied" : "e-CS not satisfied") << std::endl;
    clearBids();
    doneMatching_ = true;
    // for all unassigned person (rows)
    for (int i = 0; i < n_rows_; i++) if (!matched(leftMatch_[i]))
    {
      int j;
      double v, w;
      weight_.getBestAndSecondBest(i, j, v, w); // get the best and second best (benefit - price)
      placeBid(i, j, price_[j]+v-w+epsilon_);
      doneMatching_ = false;
    }
    // for all items, assign them to best bidder
    for (int j = 0; j < n_cols_; j++) if (winner_[j] != -1)
    {
//      std::cout << "price[" << j << "] = " << price_[j] << " --> ";
      price_[j] = bid_[j];
//      std::cout << price_[j] << std::endl;
//      std::cout << "new match " << winner_[j] << " --> " << j << std::endl;
      weight_.setPrice(j, price_[j]);
      setMatch(winner_[j], j);
    }
  }
}

template <typename W>
    void AuctionMaxWeightMatching<W>::setMatch(int new_bidder, int item)
{
  int old_bidder = rightMatch_[item];
  leftMatch_[new_bidder] = item;
  rightMatch_[item] = new_bidder;
  if (matched(old_bidder))
    unmatch(leftMatch_[old_bidder]);
}

template <typename W>
    void AuctionMaxWeightMatching<W>::clearBids()
{
  std::fill(bid_.begin(), bid_.end(), -std::numeric_limits<double>::infinity());
  std::fill(winner_.begin(), winner_.end(), -1);
}

template <typename W>
    void AuctionMaxWeightMatching<W>::placeBid(int bidder, int item, double price)
{
  if (bid_[item] < price)
  {
    bid_[item] = price;
    winner_[item] = bidder;
  }
}

template <typename W>
    void AuctionMaxWeightMatching<W>::getBestAndSecondBest(int bidder,
                                                           int &best_item,
                                                           double &best_surplus,
                                                           double &second_surplus)
{
  best_surplus = second_surplus = -std::numeric_limits<double>::infinity();
  for (int item = 0; item < n_cols_; item++)
  {
    double surplus = weight_.get(bidder, item) - price_[item];
    if (surplus > best_surplus)
    {
      best_item = item;
      second_surplus = best_surplus;
      best_surplus = surplus;
    }
    else if (surplus > second_surplus)
    {
      second_surplus = surplus;
    }
  }
}

template <typename W>
    bool AuctionMaxWeightMatching<W>::checkEpsilonComplementary() const
{
  for (int bidder = 0; bidder < n_rows_; bidder++) if (matched(leftMatch_[bidder]))
  {
    int item = leftMatch_[bidder];
    double surplus = weight_.get(bidder, item) - price_[item];
    for (int j = 0; j < n_cols_; j++) if (surplus < weight_.get(bidder, j) - price_[j] - epsilon_ - 1e-10)
    {
//      std::cout << "check bidder " << bidder << " item " << item << " "
//          << surplus << " " << weight_.get(bidder, j) - price_[j] - epsilon_ << std::endl;
      return false;
    }
  }
  return true;
}

END_ANMF_NAMESPACE;

#endif // AUCTION_MAX_WEIGHT_MATCHING_H
