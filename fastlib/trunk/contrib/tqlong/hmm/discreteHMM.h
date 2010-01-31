
#ifndef FASTLIB_DISCRETE_HMM_H
#define FASTLIB_DISCRETE_HMM_H
#include <fastlib/fastlib.h>
class DiscreteHMM {
  Matrix transition, emission;
 public:
  typedef ArrayList<index_t> StateSeq;
  typedef ArrayList<index_t> OutputSeq;

  void Generate(index_t length, OutputSeq* seq,
		StateSeq* states = NULL);
  double Decode(const OutputSeq& seq,
		Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale, 
		bool init = true);
  void Train(const ArrayList<OutputSeq>& seqs, double tolerance,
	     index_t maxIteration);
  /* Getter & setter functions */
  void LoadTransition(const char* filename);
  void LoadEmission(const char* filename);
  void Save(const char* outTR, const char* outE);
  int n_states() { return transition.n_rows(); }
  int n_symbols() { return emission.n_cols(); }
  double tr_get(index_t i, index_t j) 
    { return transition.ref(i, j); }
  double e_get(index_t i, index_t j)
    { return emission.ref(i, j); }
  static void readSEQ(TextLineReader& f, DiscreteHMM::OutputSeq* seq);
  static void readSEQs(TextLineReader& f, ArrayList<DiscreteHMM::OutputSeq>* seqs);
 private:
  void forward(const OutputSeq& seq,
	       Matrix* fs, Vector* scale);
  void backward(const OutputSeq& seq,
		Matrix* bs, Vector* scale);
  void initDecode(index_t length,
		  Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale);
  void calPStates(index_t length, Matrix* pStates, 
		  Matrix* fs, Matrix* bs);
  double calPSeq(index_t length, Vector* scale);
  void M_step(const OutputSeq& seq, const Matrix& fs, const Matrix& bs, 
	      const Vector& s, const Matrix& logTR, const Matrix&logE, 
	      Matrix* TR, Matrix* E);
  double& tr_ref(index_t i, index_t j) 
    { return transition.ref(i, j); }
  double& e_ref(index_t i, index_t j)
    { return emission.ref(i, j); }
};

#endif
