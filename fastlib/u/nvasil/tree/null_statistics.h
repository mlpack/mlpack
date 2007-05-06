/*
 * =====================================================================================
 * 
 *       Filename:  null_statistics.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  05/01/2007 02:49:14 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

class NullStatistics {
 public:	
	void Alias(const NullStatistics &other) {
		
	}
  NullStatistics &operator=(const NullStatistics &other) {
	  return *this;
	}
};


