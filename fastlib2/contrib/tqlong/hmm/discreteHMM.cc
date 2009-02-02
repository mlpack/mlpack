
#include "discreteHMM.h"
#include "support.h"

using namespace supportHMM;

index_t MaxLength(const ArrayList<DiscreteHMM::OutputSeq>& seqs);

void DiscreteHMM::LoadTransition(const char* filename) {
  data::Load(filename, &transition);
  normalizeRows(&transition);
}

void DiscreteHMM::LoadEmission(const char* filename) {
  data::Load(filename, &emission);
  normalizeRows(&emission);
}

void DiscreteHMM::Save(const char* outTR, const char* outE) {
  data::Save(outTR, transition);
  data::Save(outE, emission);
}

void DiscreteHMM::readSEQ(TextLineReader& f, DiscreteHMM::OutputSeq* seq) {
  seq->Init();
  while (f.MoreLines()) {
    ArrayList<String> strlist;
    strlist.Init();
    f.Peek().Split(", ", &strlist);
    f.Gobble();
    if (strlist.size() > 0) {
      for (index_t i = 0; i < strlist.size(); i++)
	seq->PushBackCopy(atoi(strlist[i].c_str()));
      //ot::Print(*seq);
      break;
    }
  }
}

void DiscreteHMM::readSEQs(TextLineReader& f, ArrayList<DiscreteHMM::OutputSeq>* seqs) {
  seqs->Init();
  while (1) {
    DiscreteHMM::OutputSeq seq;
    readSEQ(f, &seq);
    if (seq.size() == 0) break;
    seqs->PushBackCopy(seq);
  }
}

void DiscreteHMM::Generate(index_t length, OutputSeq* seq,
			   StateSeq* states){
  Matrix cTR, cE;

  cumulativeSum(transition, &cTR);
  cumulativeSum(emission, &cE);
	
  //ot::Print(cTR);
  //ot::Print(cE);
	
  seq->Init();
  if (states) states->Init();
	
  index_t currentState = 0;
	
  for (index_t t = 0; t < length; t++) {
    double state_rand = math::Random(0, 1);
    index_t state = 0;
    //printf("state=%d state_rand=%f rand_max=%d\n",state,state_rand, RAND_MAX);
		
    while (cTR.get(currentState, state) < state_rand)
      state++;
    //printf("state=%d\n",state);
    double output_rand = math::Random(0, 1);
    index_t output = 0;
    while (cE.get(state, output) < output_rand)
      output++;
    //printf("output=%d\n",output);
    seq->PushBackCopy(output);
    if (states) states->PushBackCopy(state);
    currentState = state;
  }
}

double DiscreteHMM::Decode(const OutputSeq& seq,
			   Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale, bool init) {
	
  if (init) initDecode(seq.size(), pStates, fs, bs, scale);
  forward(seq, fs, scale);
  backward(seq, bs, scale);
	
  calPStates(seq.size(), pStates, fs, bs);
  double logSeq = calPSeq(seq.size(), scale);
	
  return logSeq;
}

void DiscreteHMM::Train(const ArrayList<OutputSeq>& seqs, 
			double tolerance,	index_t maxIteration) {
  Matrix pStates, fs, bs;
  Vector scale;
  Matrix logTR, logE;
  Matrix TR, E;
  initDecode(MaxLength(seqs), &pStates, &fs, &bs, &scale);
  logTR.Init(n_states(), n_states());
  logE.Init(n_states(), n_symbols());
  TR.Init(n_states(), n_states());
  E.Init(n_states(), n_symbols());
	
  double loglik = 1.0;
  for (index_t iter = 0; iter < maxIteration; iter++) {
    double oldLL = loglik;
    Matrix oldTR, oldE;
    oldTR.Copy(transition);
    oldE.Copy(emission);
		
    CopyLog(transition, &logTR);
    CopyLog(emission, &logE);
    TR.SetZero();
    E.SetZero();
    loglik = 0.0;
		
    for (index_t i_seq = 0; i_seq < seqs.size(); i_seq++) {
      const OutputSeq& seq = seqs[i_seq];
      // The E-step
      loglik += Decode(seq, &pStates, &fs, &bs, &scale, false);
      // The M-step
      M_step(seq, fs, bs, scale, logTR, logE, &TR, &E);
    }
    normalizeRows(&TR);
    normalizeRows(&E);
    transition.CopyValues(TR);
    emission.CopyValues(E);
	  
    double diffTR = normDiff(oldTR, TR) / n_states();
    double diffE = normDiff(oldE, E) / n_symbols();
    double diffLOG = fabs(loglik-oldLL)/(1+fabs(oldLL));
    printf("iter = %d loglik = %f\n", iter, loglik);
    if ( diffLOG < tolerance && diffTR < tolerance && diffE < tolerance) {
      printf("Converged at iteration #%d\n", iter);
      break;
    }
  }
}

void DiscreteHMM::M_step(const OutputSeq& seq, const Matrix& fs, const Matrix& bs, 
			 const Vector& scale, const Matrix& logTR, const Matrix&logE, 
			 Matrix* TR, Matrix* E) {
  for (index_t s1 = 0; s1 < n_states(); s1++)
    for (index_t s2 = 0; s2 < n_states(); s2++)
      for (index_t i = 0; i < seq.size(); i++)
	TR->ref(s1, s2) += exp(fs.get(s1, i)+logTR.get(s1, s2)+
			       logE.get(s2, seq[i])+bs.get(s2, i+1))/scale[i+1];
  for (index_t i = 1; i < seq.size()+1; i++) {
    index_t output = seq[i-1];
    for (index_t s = 0; s < n_states(); s++)
      E->ref(s, output) += exp(fs.get(s, i)+bs.get(s, i));
  }
}

void DiscreteHMM::initDecode(index_t length,
			     Matrix* pStates, Matrix* fs, Matrix* bs, Vector* scale) {
  fs->Init(n_states(), length+1);
  scale->Init(length+1);
  bs->Init(n_states(), length+1);
  pStates->Init(n_states(), length);
}

void DiscreteHMM::forward(const OutputSeq& seq,
			  Matrix* fs, Vector* scale) {
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
      fs->ref(s, i) *= e_get(s, seq[i-1]);
      //printf("i=%d s=%d, fs(s, i) = %f\n", i,s,fs->get(s, i));
    }
    (*scale)[i] = NormalizeColumn(fs, i);
  }
}

void DiscreteHMM::backward(const OutputSeq& seq,
			   Matrix* bs, Vector* scale) {
  index_t L = seq.size()+1;
  for (index_t s = 0; s < n_states(); s++)
    bs->ref(s, L-1) = 1.0;
  for (index_t i = L-2; i >= 0; i--) 
    for (index_t s = 0; s < n_states(); s++) {
      bs->ref(s, i) = 0.0;
      for (index_t s1 = 0; s1 < n_states(); s1++)
	bs->ref(s, i) += tr_get(s, s1)*bs->get(s1, i+1)*
	  e_get(s1, seq[i]);
      bs->ref(s, i) /= (*scale)[i+1];
    }
}

void DiscreteHMM::calPStates(index_t L, Matrix* pStates, 
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

double DiscreteHMM::calPSeq(index_t length, Vector* scale) {
  double logSeq = 0.0;
  for (index_t i = 0; i < length+1; i++)
    logSeq += log((*scale)[i]);
  return logSeq;
}

index_t MaxLength(const ArrayList<DiscreteHMM::OutputSeq>& seqs) {
  index_t length = 0;
  for (index_t i = 0; i < seqs.size(); i++)
    if (length < seqs[i].size()) length = seqs[i].size();
  return length;
}
