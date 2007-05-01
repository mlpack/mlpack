/*
 * =====================================================================================
 * 
 *       Filename:  point_identity_discriminator.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  04/03/2007 06:52:25 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#ifndef POINT_IDENTITY_DISCRIMINATOR_H_
#define POINT_IDENTITY_DISCRIMINATOR_H_
#include <stdio.h>
#include "fastlib/fastlib.h"
#include "u/nvasil/timit/transcript.h"

class SimpleDiscriminator {
 public: 
	FORBID_COPY(SimpleDiscriminator);
	SimpleDiscriminator() {
	}
  inline  bool AreTheSame(index_t i, index_t j) {
	  return i==j;
	}
};

class TimitDiscriminator {
 public: 
	FORBID_COPY(TimitDiscriminator);
	enum DiscriminantType {TIMIT_SPEAKER_LOOCV=1, 
		                     TIMIT_TEST_TRAIN_LOOCV=2};
  TimitDiscriminator() {
	 
	}
	TimitDiscriminator(const DiscriminantType method,
		                 Transcript *index) {
		method_=method;
	  index_=index;   
	}
	inline bool AreTheSame(index_t id1, index_t id2) {
		if (method_==TIMIT_SPEAKER_LOOCV) {
			assert(index_[id1].speaker_!=-1);
    	assert(index_[id2].speaker_!=-1);
      return index_[id1].speaker_==index_[id2].speaker_;					
		}
		return false;
	}
 private:
	DiscriminantType method_;
	Transcript *index_;
};

#endif //POINT_IDENTITY_DISCRIMINATOR_
