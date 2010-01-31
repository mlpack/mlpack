/*
 * =====================================================================================
 * 
 *       Filename:  transcript.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  04/04/2007 10:43:37 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#ifndef  TRANSCRIPT_H_
#define  TRANSCRIPT_H_
#include <string.h>
#include <assert.h>
#include <string>
#include "base/basic_types.h"
using namespace std;

int32 Label2Speaker(const char *s);

struct Transcript {
	enum Set {TRAIN=0, TEST=1};
	enum Dialect {NewEngland=0,
		            Northern=1,
					      NorthMidland=2,
								SouthMidland=3,
							  Southern=4,
								NewYorkCity=5,
								Western=6,
								ArmyBrat=7};
	enum Speaker
  #include "speakers.h"
	enum Sentence {SA=0, SI=1, SX=2}; 
  enum Gender   {MALE=0, FEMALE=1};
	void Init(string file, uint64 point_id, uint32 time_stamp, string phoneme) {
	  point_id_=point_id;
		time_stamp_=time_stamp;
		string::size_type loc=file.find("/");
		loc=file.find("train", loc+1);
		if (loc==string::npos) {
		  loc=file.find("test", loc+1);
     	if (loc==string::npos) {
		    fprintf(stderr, "Wrong entry in transcript file\n");
			  assert(false);
	    }
			set_=TEST;
		} else {
		  set_=TRAIN;
		}
		loc=file.find("dr", loc+1);
		if (loc==string::npos) {
		  fprintf(stderr, "Wrong entry in transcript file\n");
			assert(false);
		}
		
    dialect_=Dialect(atoi(file.substr(loc+2, 1).c_str()));
    loc=file.find("/", loc+3);
	  if (loc==string::npos) {
		  fprintf(stderr, "Wrong entry in transcript file\n");
			assert(false);	
		}
		if (file.substr(loc+1, 1).c_str()=="f") {
      gender_=FEMALE;
		} else {
		  gender_=MALE;
		}
    speaker_=Speaker(Label2Speaker(file.substr(loc+2, 4).c_str())); 
    loc=loc+7;

    if (file.substr(loc,2)=="SA") {
		  sentence_=SA;
		}
		if (file.substr(loc,2)=="SX") {
		  sentence_=SX;
		}
    if (file.substr(loc,2)=="SI") {
		  sentence_=SI;
		}
   loc+=2;
	 string::size_type loc1=file.find(".", loc);
	 sentence_num_ = atoi(file.substr(loc, loc1-loc).c_str());
   strcpy(phoneme_, phoneme.c_str());

	}
  uint64   point_id_;
	uint32   time_stamp_;
  Set      set_;
	Dialect  dialect_;
	Gender   gender_;
  Speaker	 speaker_;
	Sentence sentence_;
	uint32   sentence_num_;
	char    phoneme_[5];
};
Transcript *OpenBinaryTranscriptFile(
		string binary_transcript_file);

void CloseBinaryTranscriptFile(Transcript *ptr, 
    string binary_transcript_file);



#endif // TRANSCRIPT_H_
