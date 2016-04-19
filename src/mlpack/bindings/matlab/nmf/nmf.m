function [W H] = nmf(dataPoints, rank, varargin)
%Non-negative Matrix Factorization
%
%  This program performs non-negative matrix factorization on the given dataset,
%  storing the resulting decomposed matrices in the specified files.  For an
%  input dataset V, NMF decomposes V into two matrices W and H such that
%
%  V = W * H
%
%  where all elements in W and H are non-negative.  If V is of size (n x m), then
%  W will be of size (n x r) and H will be of size (r x m), where r is the rank
%  of the factorization (specified by --rank).
%
%  Optionally, the desired update rules for each NMF iteration can be chosen from
%  the following list:
%
%   - multdist: multiplicative distance-based update rules (Lee and Seung 1999)
%   - multdiv: multiplicative divergence-based update rules (Lee and Seung 1999)
%   - als: alternating least squares update rules (Paatero and Tapper 1994)
%
%  The maximum number of iterations is specified with 'max_iterations', and the
%  minimum residue required for algorithm termination is specified with
%  'min_residue'.
%
%Parameters:
% dataPoints        - (required) Input dataset to perform NMF on.
% rank							- (required) Rank of the factorization.
% max_iterations    - (optional) Number of iterations before NMF terminates.
%																 (Default value 10000.)
% min_residue			  - (optional) The minimum root mean square residue allowed for
%                                each iteration, below which the program
%                                terminates.  Default value 1e-05.
% seed							- (optional) Random seed.If 0, 'std::time(NULL)' is used.
%														     Default 0.
% update rules			- (optional) Update rules for each iteration; ( multdist |
%                                multdiv | als ).  Default value 'multdist'.

% a parser for the inputs
p = inputParser;
p.addParamValue('max_iterations', 10000, @isscalar);
p.addParamValue('min_residue', 1e-05, @isscalar);
p.addParamValue('update_rules', 'multdist', @ischar);
p.addParamValue('seed', 0, @isscalar);

% parsing the varargin options
p.parse(varargin{:});
parsed = p.Results;

% interfacing with mlpack. transposing for machine learning standards.
[W H] = mex_nmf(dataPoints', rank, ...
	parsed.max_iterations, parsed.min_residue, ...
	parsed.update_rules, parsed.seed);
W = W';
H = H';




