/*
 * =====================================================================================
 * 
 *       Filename:  HyperRectanglePivoter.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  04/28/2007 12:03:05 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifdef HYPER_RECTANGLE_PIVOTER_H_
#define HYPER_RECTANGLE_PIVOTER_H_

#include "loki/Typelist.h"
#include "fastlib/fastlib.h"
#include "dataset/binary_dataset.h"

template<typename TYPELIST, bool diagnostic>
class HyperRectanglePivoter {
 public:
  FORBID_COPY(HyperRectanglePivoter)
	Init()	


};

#include "hyer_rectangle_pivoter_impl.h"
#endif // HYPER_RECTANGLE_PIVOTER_H_
