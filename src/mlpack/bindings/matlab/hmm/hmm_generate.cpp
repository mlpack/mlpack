#include "mex.h"

#include <mlpack/core.hpp>

#include "hmm.hpp"
#include "hmm_util.hpp"
#include <mlpack/methods/gmm/gmm.hpp>

/*
PROGRAM_INFO("Hidden Markov Model (HMM) Sequence Generator", "This "
    "utility takes an already-trained HMM (--model_file) and generates a "
    "random observation sequence and hidden state sequence based on its "
    "parameters, saving them to the specified files (--output_file and "
    "--state_file)");

PARAM_STRING_REQ("model_file", "File containing HMM (XML).", "m");
PARAM_INT_REQ("length", "Length of sequence to generate.", "l");

PARAM_INT("start_state", "Starting state of sequence.", "t", 0);
PARAM_STRING("output_file", "File to save observation sequence to.", "o",
    "output.csv");
PARAM_STRING("state_file", "File to save hidden state sequence to (may be left "
    "unspecified.", "S", "");
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
*/


using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;
using namespace mlpack::utilities;
using namespace mlpack::gmm;
using namespace mlpack::math;
using namespace arma;
using namespace std;

namespace {
	// gets the transition matrix from the struct
	void getTransition(mat & transition, const mxArray * mxarray)
	{
		mxArray * mxTransitions = mxGetField(mxarray, 0, "transition");
		if (NULL == mxTransitions)
		{
			mexErrMsgTxt("Model struct did not have transition matrix 'transition'.");
		}
		if (mxDOUBLE_CLASS != mxGetClassID(mxTransitions))
		{
			mexErrMsgTxt("Transition matrix 'transition' must have type mxDOUBLE_CLASS.");
		}
		const size_t m = mxGetM(mxTransitions);
		const size_t n = mxGetN(mxTransitions);
		transition.resize(m,n);

		double * values = mxGetPr(mxTransitions);
		for (int i = 0; i < m*n; ++i)
			transition(i) = values[i];
	}

	// writes the matlab transition matrix to the model
	template <class T>
	void writeTransition(HMM<T> & hmm, const mxArray * mxarray)
	{
		mxArray * mxTransitions = mxGetField(mxarray, 0, "transition");
		if (NULL == mxTransitions)
		{
			mexErrMsgTxt("Model struct did not have transition matrix 'transition'.");
		}
		if (mxDOUBLE_CLASS != mxGetClassID(mxTransitions))
		{
			mexErrMsgTxt("Transition matrix 'transition' must have type mxDOUBLE_CLASS.");
		}

		arma::mat transition(mxGetM(mxTransitions), mxGetN(mxTransitions));
		double * values = mxGetPr(mxTransitions);
		for (int i = 0; i < mxGetM(mxTransitions) * mxGetN(mxTransitions); ++i)
			transition(i) = values[i];

		hmm.Transition() = transition;
	}

	// argument check on the emission field
	void checkEmission(const mat & transition, const mxArray * mxarray)
	{
		if (NULL == mxarray)
		{
			mexErrMsgTxt("Model struct did not have 'emission' struct.");
		}
		if ((int) mxGetN(mxarray) != (int) transition.n_rows)
		{
			stringstream ss;
			ss << "'emissions' struct array must have dimensions 1 x "
				<<  transition.n_rows << ".";
			mexErrMsgTxt(ss.str().c_str());
		}
	}

} // closing anonymous namespace

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // argument checks
  if (nrhs != 4)
  {
    mexErrMsgTxt("Expecting four arguments.");
  }

  if (nlhs != 1)
  {
    mexErrMsgTxt("Output required.");
  }

	// seed argument
	size_t seed = (size_t) mxGetScalar(prhs[3]);

  // Set random seed.
	if (seed != 0)
    mlpack::math::RandomSeed(seed);
  else
    mlpack::math::RandomSeed((size_t) std::time(NULL));

	// length of observations
	const int length =  (int) mxGetScalar(prhs[1]);

	// start state
	const int startState = (int) mxGetScalar(prhs[2]);

  if (length <= 0)
  {
		stringstream ss;
    ss << "Invalid sequence length (" << length << "); must be greater "
        << "than or equal to 0!";
		mexErrMsgTxt(ss.str().c_str());
  }

	// getting the model type
	if (mxIsStruct(prhs[0]) == 0)
	{
		mexErrMsgTxt("Model argument is not a struct.");
	}

	mxArray * mxHmmType = mxGetField(prhs[0], 0, "hmm_type");
	if (mxHmmType == NULL)
	{
		mexErrMsgTxt("Model struct did not have 'hmm_type'.");
	}
	if (mxCHAR_CLASS != mxGetClassID(mxHmmType))
	{
		mexErrMsgTxt("'hmm_type' must have type mxCHAR_CLASS.");
	}

	// getting the model type string
	int bufLength = mxGetNumberOfElements(mxHmmType) + 1;
	char * buf;
	buf = (char *) mxCalloc(bufLength, sizeof(char));
  mxGetString(mxHmmType, buf, bufLength);
	string type(buf);
	mxFree(buf);

	cout << type << endl;

	// to be filled by the generator
	mat observations;
  Col<size_t> sequence;

	// to be removed!
	SaveRestoreUtility sr;

  if (type == "discrete")
  {
    HMM<DiscreteDistribution> hmm(1, DiscreteDistribution(1));

		// writing transition matrix to the hmm
		writeTransition(hmm, prhs[0]);

		// writing emission matrix to the hmm
		mxArray * mxEmission = mxGetField(prhs[0], 0, "emission");
		//checkEmission(hmm, mxEmission);

		vector<DiscreteDistribution> emission(hmm.Transition().n_rows);
		for (int i=0; i<hmm.Transition().n_rows; ++i)
		{
			mxArray * mxProbabilities = mxGetField(mxEmission, i, "probabilities");
			if (NULL == mxProbabilities)
			{
				mexErrMsgTxt("'probabilities' field could not be found in 'emission' struct.");
			}

			arma::vec probabilities(mxGetN(mxProbabilities));
			double * values = mxGetPr(mxProbabilities);
			for (int j=0; j<mxGetN(mxProbabilities); ++j)
				probabilities(j) = values[j];

			emission[i] = DiscreteDistribution(probabilities);
		}

		hmm.Emission() = emission;

		// At this point, the HMM model should be fully formed.
    if (startState < 0 || startState >= (int) hmm.Transition().n_rows)
    {
			stringstream ss;
      ss << "Invalid start state (" << startState << "); must be "
          << "between 0 and number of states (" << hmm.Transition().n_rows
          << ")!";
			mexErrMsgTxt(ss.str().c_str());
    }

    hmm.Generate(size_t(length), observations, sequence, size_t(startState));
  }
  else if (type == "gaussian")
  {
		/*
    //HMM<GaussianDistribution> hmm(1, GaussianDistribution(1));

		// get transition matrix
		//mat transition;
		//getTransition(transition, prhs[0]);

		//hmm.Transition() = transition;
		//cout << transition << endl;
		arma::mat transition("0.75 0.25; 0.25 0.75");

		// get emission
		//vector<GaussianDistribution> emission(transition.n_rows);
		vector<GaussianDistribution> emission;
  	GaussianDistribution g1("5.0 5.0", "1.0 0.0; 0.0 1.0");
  	GaussianDistribution g2("-5.0 -5.0", "1.0 0.0; 0.0 1.0");
  	emission.push_back(g1);
  	emission.push_back(g2);


		//HMM<GaussianDistribution> hmm(transition, emission);
		//hmm.Emission() = emission;
		HMM<GaussianDistribution> hmm(transition, emission);
		*/

		// Our distribution will have three two-dimensional output Gaussians.
		cout << "following the test" << endl;
  	HMM<GaussianDistribution> hmm(3, GaussianDistribution(2));
  	hmm.Transition() = arma::mat("0.4 0.6 0.8; 0.2 0.2 0.1; 0.4 0.2 0.1");
  	hmm.Emission()[0] = GaussianDistribution("0.0 0.0", "1.0 0.0; 0.0 1.0");
  	hmm.Emission()[1] = GaussianDistribution("2.0 2.0", "1.0 0.5; 0.5 1.2");
  	hmm.Emission()[2] = GaussianDistribution("-2.0 1.0", "2.0 0.1; 0.1 1.0");

  	// Now we will generate a long sequence.
  	std::vector<arma::mat> observations2(1);
  	std::vector<arma::Col<size_t> > states2(1);

		// testing
  	SaveHMM(hmm, sr);
  	sr.WriteFile("testMexGaussian.xml");

  	// Start in state 1 (no reason).
  	cout << "test generation" << endl;
		hmm.Generate(10000, observations2[0], states2[0], 1);
		cout << "test complete" << endl;

    if (startState < 0 || startState >= (int) hmm.Transition().n_rows)
    {
			stringstream ss;
			ss << "Invalid start state (" << startState << "); must be "
          << "between 0 and number of states (" << hmm.Transition().n_rows
          << ")!";
			mexErrMsgTxt(ss.str().c_str());
    }
		cout << "generating!" << endl;
    hmm.Generate(size_t(length), observations, sequence, size_t(startState));
		cout << "done!" << endl;
  }
  else if (type == "gmm")
  {
    HMM<GMM<> > hmm(1, GMM<>(1, 1));

    LoadHMM(hmm, sr);

    if (startState < 0 || startState >= (int) hmm.Transition().n_rows)
    {
      Log::Fatal << "Invalid start state (" << startState << "); must be "
          << "between 0 and number of states (" << hmm.Transition().n_rows
          << ")!" << endl;
    }

    hmm.Generate(size_t(length), observations, sequence, size_t(startState));
  }
  else
  {
    Log::Fatal << "Unknown HMM type '" << type << "'" << "'!" << endl;
  }

	cout << "returning to matlab" << endl;

	// Setting values to be returned to matlab
	mwSize ndim = 1;
  mwSize dims[1] = {1};
  const char * fieldNames[2] = {
    "observations"
    , "states"
  };

	plhs[0] = mxCreateStructArray(ndim, dims, 2, fieldNames);

	mxArray * tmp;
	double * values;

	cout << observations.n_rows << "," << observations.n_cols << endl;
	cout << sequence.n_rows << "," << sequence.n_cols << endl;
	cout << observations << endl;
	cout << sequence << endl;

	// settings the observations
	tmp = mxCreateDoubleMatrix(observations.n_rows, observations.n_cols, mxREAL);
	values = mxGetPr(tmp);
	for (int i=0; i<observations.n_rows * observations.n_cols; ++i)
		values[i] = observations(i);

	// note: SetField does not copy the data structure.
	// mxDuplicateArray does the necessary copying.
	mxSetFieldByNumber(plhs[0], 0, 0, mxDuplicateArray(tmp));
	mxDestroyArray(tmp);

	// settings the observations
	tmp = mxCreateDoubleMatrix(sequence.n_rows, sequence.n_cols, mxREAL);
	values = mxGetPr(tmp);
	for (int i=0; i<length; ++i)
		values[i] = sequence(i);

	// note: SetField does not copy the data structure.
	// mxDuplicateArray does the necessary copying.
	mxSetFieldByNumber(plhs[0], 0, 1, mxDuplicateArray(tmp));
	mxDestroyArray(tmp);
}

		/*
		mxArray * mxEmission = mxGetField(prhs[0], 0, "emission");
		checkEmission(transition, mxEmission);

		vector<GaussianDistribution> emission(transition.n_rows);
		for (int i=0; i<transition.n_rows; ++i)
		{
			// mean
			mxArray * mxMean = mxGetField(mxEmission, i, "mean");
			if (NULL == mxMean)
			{
				mexErrMsgTxt("'mean' field could not be found in 'emission' struct.");
			}

			arma::vec mean(mxGetN(mxMean));
			double * values = mxGetPr(mxMean);
			for (int j=0; j<mxGetN(mxMean); ++j)
				mean(j) = values[j];

			cout << mean << endl;

			// covariance
			mxArray * mxCovariance = mxGetField(mxEmission, i, "covariance");
			if (NULL == mxCovariance)
			{
				mexErrMsgTxt("'covariance' field could not be found in 'emission' struct.");
			}

			const size_t m = (size_t) mxGetM(mxCovariance);
			const size_t n = (size_t) mxGetN(mxCovariance);
			mat covariance(m, n);
			values = mxGetPr(mxCovariance);
			for (int j=0; j < m * n; ++j)
				covariance(j) = values[j];

			cout << covariance << endl;

			emission[i] = GaussianDistribution(mean, covariance);
		}
		*/
