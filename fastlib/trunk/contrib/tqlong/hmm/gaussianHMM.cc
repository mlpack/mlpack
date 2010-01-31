
#include "gaussianHMM.h"
#include "support.h"

using namespace supportHMM;

index_t MaxLength(const ArrayList<GaussianHMM::OutputSeq>& seqs);

void GaussianHMM::LoadTransition(const char* filename) {
  data::Load(filename, &transition);
  normalizeRows(&transition);
}

void GaussianHMM::LoadEmission(const char* filename) {
  Matrix tmp;
  data::Load(filename, &tmp);
  index_t dim = tmp.n_rows();
  index_t n = tmp.n_cols()/(dim+1);
  emission.Init();
  for (index_t i = 0; i < n; i++) {
    GaussianDistribution gd(dim);
    GaussianDistribution::createFromCols(tmp, i*(dim+1), &gd);
    emission.PushBackCopy(gd);
  }
}

void GaussianHMM::Save(const char* outTR, const char* outE) {
  data::Save(outTR, transition);
  FILE* f = fopen(outE, "w");
  for (index_t i = 0; i < n_states(); i++)
    emission[i].Save(f);
  fclose(f);
}

void GaussianHMM::InitRandom(index_t dim, index_t n_state) {
  RandomInit(n_state, n_state, &transition);
  normalizeRows(&transition);
  emission.Init();
  for (index_t i = 0; i < n_state; i++) {
    GaussianDistribution gd(dim);
    emission.PushBackCopy(gd);
  }
}

void GaussianHMM::printSEQ(FILE* f, const OutputSeq& seq) {
  fprintf(f, "%d\n", seq.size());
  for (index_t i = 0; i < seq.size(); i++) 
    printVector(f, seq[i]);
}

void GaussianHMM::readSEQ(TextLineReader& f, OutputSeq* seq) {
  seq->Init();
  while (f.MoreLines()) {
    ArrayList<String> strlist;
    strlist.Init();
    f.Peek().Split(", ", &strlist);
    f.Gobble();
    // is number of vectors
    if (strlist.size() != 1) continue;
    index_t length = atoi(strlist[0].c_str());
    //printf("length=%d\n", length);
    if (length == 0) continue;
    for (index_t i = 0; i < length; i++) {
      ArrayList<String> strlist;
      strlist.Init();
      f.Peek().Split(", ", &strlist);
      f.Gobble();
      Vector tmp;
      tmp.Init(strlist.size());
      for (index_t i = 0; i < strlist.size(); i++)
	tmp[i] = atof(strlist[i].c_str());
      seq->PushBackCopy(tmp);
      //ot::Print(tmp);
    }
    break;
  }
}

void GaussianHMM::readSEQs(TextLineReader& f, ArrayList<OutputSeq>* seq) {
  seq->Init();
  while (1) {
    OutputSeq tmp;
    readSEQ(f, &tmp);
    if (tmp.size() == 0) break;
    seq->PushBackCopy(tmp);
  }
}

void GaussianHMM::Generate(index_t length, OutputSeq* seq,
			   StateSeq* states){
  Matrix cTR;
  cumulativeSum(transition, &cTR);
  seq->Init();
  if (states) states->Init();
  index_t currentState = 0;
  for (index_t t = 0; t < length; t++) {
    double state_rand = math::Random(0, 1);
    index_t state = 0;
    while (cTR.get(currentState, state) < state_rand)
      state++;
    Vector output;
    emission[state].Generate(&output);
    seq->PushBackCopy(output);
    if (states) states->PushBackCopy(state);
    currentState = state;
  }
}

double GaussianHMM::Decode(const OutputSeq& seq,
			   Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale, Matrix* pOutput,
			   bool init) {
	
  if (init) initDecode(seq.size(), pStates, fs, bs, scale, pOutput);
  //for (int i = 0; i < n_states(); i++)
  //	emission[i].Save(stdout);
  calPOutput(seq, pOutput);
  //ot::Print(*pOutput);
  forward(seq, fs, scale, pOutput);
  //ot::Print(*fs);
  //ot::Print(*scale);
  backward(seq, bs, scale, pOutput);
  //ot::Print(*bs);
	
  calPStates(seq.size(), pStates, fs, bs);
  double logSeq = calPSeq(seq.size(), scale);
	
  return logSeq;
}

void GaussianHMM::Train(const ArrayList<OutputSeq>& seqs, 
			double tolerance,	index_t maxIteration) {
  Matrix pStates, fs, bs, pOutput;
  Vector scale;
  Matrix logTR;
  Matrix TR;
  initDecode(MaxLength(seqs), &pStates, &fs, &bs, &scale, &pOutput);
  logTR.Init(n_states(), n_states());
  TR.Init(n_states(), n_states());
	
  double loglik = 1.0;
  for (index_t iter = 0; iter < maxIteration; iter++) {
    double oldLL = loglik;
    Matrix oldTR;
    ArrayList<GaussianDistribution> oldE;
    oldTR.Copy(transition);
    oldE.InitCopy(emission);
		
    CopyLog(transition, &logTR);
    TR.SetZero();
    for (index_t s = 0; s < n_states(); s++) 
      emission[s].StartAccumulate();
    loglik = 0.0;
		
    for (index_t i_seq = 0; i_seq < seqs.size(); i_seq++) {
      const OutputSeq& seq = seqs[i_seq];
      // The E-step
      loglik += Decode(seq, &pStates, &fs, &bs, &scale, &pOutput, false);
      // The M-step
      M_step(seq, fs, bs, scale, logTR, pOutput, &TR, &emission);
    }
    normalizeRows(&TR);
    transition.CopyValues(TR);
    for (index_t s = 0; s < n_states(); s++) 
      emission[s].EndAccumulate();
	  
    double diffTR = normDiff(oldTR, TR) / n_states();
    //double diffE = normDiff(oldE, E) / n_states();
    double diffLOG = fabs(loglik-oldLL)/(1+fabs(oldLL));
    printf("iter = %d loglik = %f\n", iter, loglik);
    if ( diffLOG < tolerance && diffTR < tolerance /*&& diffE < tolerance*/) {
      printf("Converged at iteration #%d\n", iter);
      break;
    }
  }
}

void GaussianHMM::calPOutput(const OutputSeq& seq, Matrix* pOutput) {
  for (index_t i = 0; i < seq.size(); i++) {
    for (index_t s = 0; s < n_states(); s++) {
      pOutput->ref(s, i) = emission[s].logP(seq[i]);
      //printf("%10f,", pOutput->ref(s, i));
    }
    //printf("\n");
  }
}

void GaussianHMM::M_step(const OutputSeq& seq, const Matrix& fs, const Matrix& bs, 
			 const Vector& scale, const Matrix& logTR, const Matrix& pOutput, 
			 Matrix* TR, ArrayList<GaussianDistribution>* E) {
  for (index_t s1 = 0; s1 < n_states(); s1++)
    for (index_t s2 = 0; s2 < n_states(); s2++)
      for (index_t i = 0; i < seq.size(); i++)
	TR->ref(s1, s2) += exp(fs.get(s1, i)+logTR.get(s1, s2)+
			       pOutput.get(s2, i)+bs.get(s2, i+1))/scale[i+1];
  for (index_t i = 1; i < seq.size()+1; i++) {
    const Vector& output = seq[i-1];
    for (index_t s = 0; s < n_states(); s++)
      (*E)[s].Accumulate(output, exp(fs.get(s, i)+bs.get(s, i)));
  }
}

void GaussianHMM::initDecode(index_t length,
			     Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale, Matrix* pOutput) {
  fs->Init(n_states(), length+1);
  scale->Init(length+1);
  bs->Init(n_states(), length+1);
  pStates->Init(n_states(), length);
  pOutput->Init(n_states(), length);
}

void GaussianHMM::forward(const OutputSeq& seq,
			  Matrix* fs, Vector* scale, Matrix* pOutput) {
  for (index_t s = 0; s < n_states(); s++)
    fs->ref(s, 0) = 0.0;
  fs->ref(0, 0) = 1.0;
  (*scale)[0] = 1.0;
  index_t L = seq.size()+1;
  for (index_t i = 1; i < L; i++) {
    for (index_t s = 0; s < n_states(); s++) {
      fs->ref(s, i) = 0;
      for (index_t s1 = 0; s1 < n_states(); s1++)
	fs->ref(s, i) +=  fs->get(s1, i-1)*tr_get(s1, s);
      fs->ref(s, i) = exp(log(fs->get(s, i)) + pOutput->get(s, i-1));
      //printf("i=%d s=%d, fs(s, i) = %f\n", i,s,fs->get(s, i));
    }
    (*scale)[i] = NormalizeColumn(fs, i);
  }
}

void GaussianHMM::backward(const OutputSeq& seq,
			   Matrix* bs, Vector* scale, Matrix* pOutput) {
  index_t L = seq.size()+1;
  for (index_t s = 0; s < n_states(); s++)
    bs->ref(s, L-1) = 1.0;
  for (index_t i = L-2; i >= 0; i--) 
    for (index_t s = 0; s < n_states(); s++) {
      bs->ref(s, i) = 0.0;
      for (index_t s1 = 0; s1 < n_states(); s1++)
	bs->ref(s, i) += exp(log(tr_get(s, s1))+log(bs->get(s1, i+1))+
			     pOutput->get(s1, i));
      bs->ref(s, i) /= (*scale)[i+1];
    }
}

void GaussianHMM::calPStates(index_t L, Matrix* pStates, 
			     Matrix* fs, Matrix* bs) {
  for (index_t i = 0; i < L; i++)
    for (index_t s = 0; s < n_states(); s++)
      pStates->ref(s, i) = fs->get(s, i+1) * bs->get(s, i+1);	
  //ot::Print(*fs);
  for (index_t i = 0; i < L+1; i++) 
    for (index_t s = 0; s < n_states(); s++) {
      fs->ref(s, i) = log(fs->get(s, i));
      bs->ref(s, i) = log(bs->get(s, i));
    }
}

double GaussianHMM::calPSeq(index_t length, Vector* scale) {
  double logSeq = 0.0;
  for (index_t i = 0; i < length+1; i++)
    logSeq += log((*scale)[i]);
  return logSeq;
}

index_t MaxLength(const ArrayList<GaussianHMM::OutputSeq>& seqs) {
  index_t length = 0;
  for (index_t i = 0; i < seqs.size(); i++)
    if (length < seqs[i].size()) length = seqs[i].size();
  return length;
}
