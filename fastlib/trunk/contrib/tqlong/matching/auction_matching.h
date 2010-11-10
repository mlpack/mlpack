#ifndef AUCTION_MATCHING_H
#define AUCTION_MATCHING_H

#include <algorithm>
#include "matching.h"

MATCHING_NAMESPACE_BEGIN;

template <typename W>
    class AuctionMatching
{
public:
  typedef W                weight_type;
protected:
  weight_type&        weight_;
  double              epsilon_;
  std::vector<int>    left_, right_, bidders_;
  std::vector<double> bids_;

  void clearMatches();
  void forwardAuction(double &pruned, double &total);
  void placeBid(int l, int r, double price);
  void setMatch(int l, int r, double price);
public:
  AuctionMatching(weight_type& weight);
  void doMatch();
  int n_left() const { return weight_.n_rows(); }
  int n_right() const { return weight_.n_cols(); }
  int left(int index) const { return left_.at(index); }
  int right(int index) const { return right_.at(index); }
  double getP(int l, int r) { return weight_.get(l, r)-weight_.price(r); }
  double price(int r) { return weight_.price(r); }
};

template <typename W>
    AuctionMatching<W>::AuctionMatching(weight_type &weight)
      : weight_(weight), left_(n_left()), right_(n_right()),
        bidders_(n_right()), bids_(n_right())
{
}

template <typename W>
    void AuctionMatching<W>::doMatch()
{
  // epsilon scaling
  double total_pruned = 0, total_cals = 0;
  for (epsilon_ = 1.0/n_left(); epsilon_ >= 1.0/n_left(); epsilon_ /= 2)
  {
    double pruned, total;
    clearMatches();
    forwardAuction(pruned, total);
    total_pruned += pruned;
    total_cals += total;
    std::cout << "epsilon = " << epsilon_ << " cals = " << total_pruned << "/" << total_cals << "\n";
  }
}

template <typename W>
    void AuctionMatching<W>::clearMatches()
{
  std::fill(left_.begin(), left_.end(), -1);
  std::fill(right_.begin(), right_.end(), -1);
}

template <typename W>
    void AuctionMatching<W>::forwardAuction(double &pruned, double &total)
{
  pruned = total = 0;
  while (1)
  {
    bool doneMatching = true;
    std::fill(bids_.begin(), bids_.end(), -std::numeric_limits<double>::infinity());
    std::fill(bidders_.begin(), bidders_.end(), -1);
    weight_.refresh();
    for (int i = 0; i < n_left(); i++) if (left(i) == -1)
    {
      doneMatching = false;
      std::vector<int> bests(2, -1);
      pruned += weight_.kBest(i, bests);
      total += n_right();
      if (bests[0] == -1 || bests[1] == -1)
      {
        printf("error kBest\n");
        return;
      }
      double v = getP(i, bests[0]), w = getP(i, bests[1]);
      placeBid(i, bests[0], price(bests[0])+v-w+epsilon_);
    }
    if (doneMatching) break;
    for (int j = 0; j < n_right(); j++) if (bidders_[j] != -1)
    {
      setMatch(bidders_[j], j, bids_[j]);
    }
  }
}

template <typename W>
    void AuctionMatching<W>::placeBid(int l, int r, double price)
{
  if (bids_[r] < price)
  {
    bids_[r] = price;
    bidders_[r] = l;
  }
}

template <typename W>
    void AuctionMatching<W>::setMatch(int l, int r, double price)
{
  weight_.setPrice(r, price);
  int ol = right(r);
  left_[l] = r;
  right_[r] = l;
  if (ol != -1)
    left_[ol] = -1;
}

MATCHING_NAMESPACE_END;

#endif // AUCTION_MATCHING_H
