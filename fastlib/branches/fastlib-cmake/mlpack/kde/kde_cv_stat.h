/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
#ifndef KDE_CV_STAT_H
#define KDE_CV_STAT_H

template<typename TKernel>
class VKdeCVStat {
 public:

  /** @brief The minimum bandwidth among the points owned by this
   *         node.
   */
  TKernel min_bandwidth_kernel_;

  /** @brief The maximum bandwidth among the points owned by this
   *         node.
   */
  TKernel max_bandwidth_kernel_;

  /** @brief The weight sum of the points owned by this node.
   */
  double weight_sum_;

  /** @brief Gets the weight sum.
   */
  double get_weight_sum() {
    return weight_sum_;
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const VKdeCVStat& left_stat, const VKdeCVStat& right_stat) {
  }
  
  VKdeCVStat() {
  }
    
  ~VKdeCVStat() {
  }
    
};

template<typename TKernelAux>
class KdeCVStat {
 public:

  /** @brief The far field expansion created by the reference points
   *         in this node.
   */
  typename TKernelAux::TFarFieldExpansion first_farfield_expansion_;
  
  /** @brief The far field expansion
   */
  typename TKernelAux::TFarFieldExpansion second_farfield_expansion_;

  /** @brief Gets the weight sum.
   */
  double get_weight_sum() {
    return first_farfield_expansion_.get_weight_sum();
  }
    
  void Init(const TKernelAux &first_ka, const TKernelAux &second_ka) {
    first_farfield_expansion_.Init(first_ka);
    second_farfield_expansion_.Init(second_ka);
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count) {
  }
    
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const KdeCVStat& left_stat, const KdeCVStat& right_stat) {
  }
    
  void Init(const Vector& center, const TKernelAux &first_ka,
	    const TKernelAux &second_ka) {
    first_farfield_expansion_.Init(center, first_ka);
    second_farfield_expansion_.Init(center, second_ka);
  }
    
  KdeCVStat() { }
    
  ~KdeCVStat() { }
    
};

#endif
