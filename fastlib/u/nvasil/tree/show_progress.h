/*
 * =====================================================================================
 * 
 *       Filename:  show_progress.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  02/27/2007 05:57:50 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#include <stdio.h>
#include "base/basic_types.h"

class ShowProgress {
 private:
  int32 current_percentage_;
 public:	
  ShowProgress() {
	  current_percentage_=0;
	}
	void Reset() {
	   printf("00%%");
	}
	void Show(uint64 i, uint64 total) {
    int32 percentage = (int32)(i*100.0 / total);
	  if (percentage != current_percentage_) {   
      printf("\b\b\b%2d%%", percentage);
  	  fflush(stdout);
  	  current_percentage_ = percentage;
    }
	}
};


