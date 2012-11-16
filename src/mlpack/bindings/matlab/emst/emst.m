function result = emst(dataPoints, varargin)
% result = emst(dataPoints, varargin)
%
% Compute the Euclidean minimum spanning tree of a set of input points using the
% dual-tree Boruvka algorithm.
%
% The output is saved in a three-column matrix, where each row indicates an
% edge.  The first column corresponds to the lesser index of the edge; the
% second column corresponds to the greater index of the edge; and the third
% column corresponds to the distance between the two points.
%
% Required parameters:
%
% dataPoints - The matrix of data points. Columns are assumed to represent
%              dimensions, with rows representing separate points.
%
% Optional parameters (i.e. emst(..., 'parameter', value, ...)):
%
% 'method'   - The algorithm for computing the tree. 'naive' or 'boruvka', with
%              'boruvka' being the default dual-tree Boruvka algorithm.
% 'leafSize' - Leaf size in the kd-tree.  One-element leaves give the
%              empirically best performance, but at the cost of greater memory
%              requirements.  Defaults to 1.
%
% Examples:
%
% result = emst(dataPoints);
% result = emst(dataPoints, 'method', 'naive');
% result = emst(dataPoints, 'method', 'naive', 'leafSize', 5);

% A parser for the inputs.
p = inputParser;
p.addParamValue('method', 'boruvka', ...
    @(x) strcmpi(x, 'naive') || strcmpi(x, 'boruvka'));
p.addParamValue('leafSize', 1, @isscalar);

% Parse the varargin options.
p.parse(varargin{:});
parsed = p.Results;

% Interface with mlpack. Transpose to machine learning standards.  MLPACK
% expects column-major matrices; the user has passed in a row-major matrix.
if strcmpi(parsed.method, 'boruvka')
  result = emst_mex(dataPoints', 1, parsed.leafSize);
    result = result';
  return;
else
  result = emst_mex(dataPoints', 0, 1);
    result = result';
  return;
end

