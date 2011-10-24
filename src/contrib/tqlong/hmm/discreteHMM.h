
#ifndef FASTLIB_DISCRETE_HMM_H
#define FASTLIB_DISCRETE_HMM_H
#include <fastlib/fastlib.h>
class DiscreteHMM {
  Matrix transition, emission;
 public:
  typedef ArrayList<size_t> StateSeq;
  typedef ArrayList<size_t> OutputSeq;

  void Generate(size_t length, OutputSeq* seq,
		StateSeq* states = NULL);
  double Decode(const OutputSeq& seq,
		Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale, 
		bool init = true);
  void Train(const ArrayList<OutputSeq>& seqs, double tolerance,
	     size_t maxIteration);
  /* Getter & setter functions */
  void LoadTransition(const char* filename);
  void LoadEmission(const char* filename);
  void Save(const char* outTR, const char* outE);
  int n_states() { return transition.n_rows(); }
  int n_symbols() { return emission.n_cols(); }
  double tr_get(size_t i, size_t j) 
    { return transition.ref(i, j); }
  double e_get(size_t i, size_t j)
    { return emission.ref(i, j); }
  static void readSEQ(TextLineReader& f, DiscreteHMM::OutputSeq* seq);
  static void readSEQs(TextLineReader& f, ArrayList<DiscreteHMM::OutputSeq>* seqs);
 private:
  void forward(const OutputSeq& seq,
	       Matrix* fs, Vector* scale);
  void backward(const OutputSeq& seq,
		Matrix* bs, Vector* scale);
  void initDecode(size_t length,
		  Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale);
  void calPStates(size_t length, Matrix* pStates, 
		  Matrix* fs, Matrix* bs);
  double calPSeq(size_t length, Vector* scale);
  void M_step(const OutputSeq& seq, const Matrix& fs, const Matrix& bs, 
	      const Vector& s, const Matrix& logTR, const Matrix&logE, 
	      Matrix* TR, Matrix* E);
  double& tr_ref(size_t i, size_t j) 
    { return transition.ref(i, j); }
  double& e_ref(size_t i, size_t j)
    { return emission.ref(i, j); }
};

#endif
