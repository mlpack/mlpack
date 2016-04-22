function sequence = hmm_generate(model, sequence_length, varargin)
%Hidden Markov Model (HMM) Sequence Generator
%
%  This utility takes an already-trained HMM (model) and generates a
%  random observation sequence and hidden state sequence based on its parameters,
%  saving them to the specified files (--output_file and --state_file)
%
%Parameters:
% model           - (required) HMM model struct.
% sequence_length - (required) Length of the sequence to produce.
% start_state	    - (optional) Starting state of sequence.  Default value 0.
% seed            - (optional) Random seed.  If 0, 'std::time(NULL)' is used.
%                   Default value 0.

% a parser for the inputs
p = inputParser;
p.addParamValue('start_state', 0, @isscalar);
p.addParamValue('seed', 0, @isscalar);

% parsing the varargin options
p.parse(varargin{:});
parsed = p.Results;

% interfacing with mlpack.
sequence = mex_hmm_generate(model, sequence_length, ...
	parsed.start_state, parsed.seed);


