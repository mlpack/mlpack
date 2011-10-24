
#ifndef FASTLIB_GAUSSIAN_HMM_H
#define FASTLIB_GAUSSIAN_HMM_H
#include <fastlib/fastlib.h>
#include "gaussianDistribution.h"

class GaussianHMM {
  Matrix transition;
  ArrayList<GaussianDistribution> emission;
 public:
  typedef ArrayList<size_t> StateSeq;
  typedef ArrayList<Vector> OutputSeq;

  void Generate(size_t length, OutputSeq* seq,
		StateSeq* states = NULL);
  double Decode(const OutputSeq& seq,
		Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale, Matrix* pOutput,
		bool init = true);
  void Train(const ArrayList<OutputSeq>& seqs, double tolerance,
	     size_t maxIteration);
  /* Getter & setter functions */
  void LoadTransition(const char* filename);
  void LoadEmission(const char* filename);
  void Save(const char* outTR, const char* outE);
  void InitRandom(size_t dim, size_t n_state);
  int n_states() { return transition.n_rows(); }
  int n_dim() { return emission[0].n_dim(); }
  double tr_get(size_t i, size_t j) 
    { return transition.ref(i, j); }
  const GaussianDistribution& e_get(size_t i)
    { return emission[i]; }
  static void printSEQ(FILE* f, const OutputSeq& seq);
  static void readSEQ(TextLineReader& f, OutputSeq* seq);
  static void readSEQs(TextLineReader& f, ArrayList<OutputSeq>* seq);
 private:
  void calPOutput(const OutputSeq& seq, Matrix* pOutput);
  void forward(const OutputSeq& seq,
	       Matrix* fs, Vector* scale, Matrix* pOutput);
  void backward(const OutputSeq& seq,
		Matrix* bs, Vector* scale, Matrix* pOutput);
  void initDecode(size_t length,
		  Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale, Matrix* pOutput);
  void calPStates(size_t length, Matrix* pStates, 
		  Matrix* fs, Matrix* bs);
  double calPSeq(size_t length, Vector* scale);
  void M_step(const OutputSeq& seq, const Matrix& fs, const Matrix& bs, 
	      const Vector& s, const Matrix& logTR, const Matrix& pOutput, 
	      Matrix* TR, ArrayList<GaussianDistribution>* E);
  double& tr_ref(size_t i, size_t j) 
    { return transition.ref(i, j); }
  GaussianDistribution& e_ref(size_t i)
    { return emission[i]; }
};

#endif
