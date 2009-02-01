#include <fastlib/fastlib.h>
#include "discreteHMM.h"

const fx_entry_doc hmm_train_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"fileTR", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM transition.\n"},
  {"fileE", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM emission.\n"},
  {"fileSEQ", FX_PARAM, FX_STR, NULL,
   "  Input file for the sequences.\n"},
  {"outTR", FX_PARAM, FX_STR, NULL,
   "  Output file for the transition.\n"},
  {"outE", FX_PARAM, FX_STR, NULL,
   "  Output file for the emission.\n"},
  {"tol", FX_PARAM, FX_DOUBLE, NULL,
   "  Error tolerance, default = 1e-5.\n"},
  {"maxiter", FX_PARAM, FX_INT, NULL,
   "  Maximum number of iterations, default = 500.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_train_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_train_main_doc = {
  hmm_train_main_entries, hmm_train_main_submodules,
  "This is a program generating sequences from HMM models.\n"
};

void readSEQ(TextLineReader& f, DiscreteHMM::OutputSeq* seq) {
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

void readSEQs(TextLineReader& f, ArrayList<DiscreteHMM::OutputSeq>* seqs) {
	seqs->Init();
	while (1) {
  	DiscreteHMM::OutputSeq seq;
  	readSEQ(f, &seq);
		if (seq.size() == 0) break;
		seqs->PushBackCopy(seq);
	}
}

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_train_main_doc );
  const char* type = fx_param_str_req(fx_root, "type");
  const char* fileTR = fx_param_str_req(fx_root, "fileTR");
  const char* fileE = fx_param_str_req(fx_root, "fileE");
  const char* fileSEQ = fx_param_str(fx_root, "fileSEQ", "seq.out");
  const char* outTR = fx_param_str(fx_root, "outTR", "tr.out");
  const char* outE = fx_param_str(fx_root, "outE", "e.out");
  double tolerance = fx_param_double(fx_root, "tol", 1e-5);
  double maxIteration = fx_param_int(fx_root, "maxiter", 500);
  
  TextLineReader f;
  f.Open(fileSEQ);
  if (strcmp(type, "discrete") == 0) {
  	DiscreteHMM hmm;
  	hmm.LoadTransition(fileTR);
  	hmm.LoadEmission(fileE);
  	ArrayList<DiscreteHMM::OutputSeq> seqs;
  	readSEQs(f, &seqs);
  	
  	hmm.Train(seqs, tolerance, maxIteration);
  	hmm.Save(outTR, outE);
	}
  f.Close();
  
  fx_done(NULL);
}

