
#ifndef FASTLIB_GAUSSIAN_HMM_H
#define FASTLIB_GAUSSIAN_HMM_H
#include <fastlib/fastlib.h>

class GaussianDistribution {
	Vector mean;
	Matrix covariance;
	
	double gConst;
	Matrix invCov;
	Matrix sqrCov;
	
	Vector accMean;
	Matrix accCov;
	double accDenom;
public:
	GaussianDistribution(const Vector& mean, const Matrix& cov);
	GaussianDistribution(index_t dim = 1);
	GaussianDistribution(const GaussianDistribution& gd);
	
	double logP(const Vector& x);	
	static void createFromCols(const Matrix& src,
		index_t col, GaussianDistribution* tmp);
	void Generate(Vector* x);
	void StartAccumulate();
	void EndAccumulate();
	void Accumulate(const Vector& x, double weight);
			
	void Save(FILE* f);	
	const Vector& getMean() { return mean; }
	const Matrix& getCov() { return covariance; }
	void InitMeanCov(index_t dim);
	void setMeanCov(const Vector& mean, const Matrix& cov);
	index_t n_dim() { return mean.length(); }
};

class GaussianHMM {
	Matrix transition;
	ArrayList<GaussianDistribution> emission;
public:
	typedef ArrayList<index_t> StateSeq;
	typedef ArrayList<Vector> OutputSeq;

	void Generate(index_t length, OutputSeq* seq,
		StateSeq* states = NULL);
	double Decode(const OutputSeq& seq,
		Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale, Matrix* pOutput,
		bool init = true);
	void Train(const ArrayList<OutputSeq>& seqs, double tolerance,
		index_t maxIteration);
/* Getter & setter functions */
	void LoadTransition(const char* filename);
	void LoadEmission(const char* filename);
	void Save(const char* outTR, const char* outE);
	int n_states() { return transition.n_rows(); }
	int n_dim() { return emission[0].n_dim(); }
	double tr_get(index_t i, index_t j) 
		{ return transition.ref(i, j); }
	const GaussianDistribution& e_get(index_t i)
		{ return emission[i]; }
	static void printSEQ(FILE* f, const OutputSeq& seq);
private:
	void calPOutput(const OutputSeq& seq, Matrix* pOutput);
	void forward(const OutputSeq& seq,
		Matrix* fs, Vector* scale, Matrix* pOutput);
	void backward(const OutputSeq& seq,
		Matrix* bs, Vector* scale, Matrix* pOutput);
	void initDecode(index_t length,
		Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale, Matrix* pOutput);
	void calPStates(index_t length, Matrix* pStates, 
	  Matrix* fs, Matrix* bs);
	double calPSeq(index_t length, Vector* scale);
	void M_step(const OutputSeq& seq, const Matrix& fs, const Matrix& bs, 
		const Vector& s, const Matrix& logTR, const Matrix& pOutput, 
		Matrix* TR, ArrayList<GaussianDistribution>* E);
	double& tr_ref(index_t i, index_t j) 
		{ return transition.ref(i, j); }
	GaussianDistribution& e_ref(index_t i)
		{ return emission[i]; }
};

#endif
