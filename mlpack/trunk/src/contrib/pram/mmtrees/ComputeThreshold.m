function [gamma] = ComputeThreshold(UqSizes, alpha)
% UqSizes should be a column of the values
% 0 < alpha <= 1
%
% The gamma value is computed for which
% gamma = arg min_gamma E_q [gamma - |Uq|]_+
% subject to E_q [I(|Uq| > gamma)] <= alpha
%
% Since there are only finitely many values of gamma
% possible, the optimization will be very stupid.

g_min = min(UqSizes);
g_max = max(UqSizes);
nQs = size(UqSizes, 1);

gamma = g_max;

% E_q [gamma - |Uq|]_+
objFun = ObjFun(UqSizes, gamma);

% E_q [ I(|Uq| > gamma) ] 
constrFun = ConstrFun(UqSizes, gamma);


display(sprintf('Starting values:Gmin:%d, Gmax:%d', g_min, ...
		g_max));
display(sprintf('Obj.Fun.: %f, Constr.Fun: %f\n', objFun, ...
		constrFun));

minObjFun = objFun;
minGamma = gamma;

while constrFun <= alpha
  
  if (minObjFun > objFun) 
    minObjFun = objFun;
    minGamma = gamma;
  end

  gamma = gamma - 1;
  objFun = ObjFun(UqSizes, gamma);
  constrFun = ConstrFun(UqSizes, gamma);
  
  
end

if gamma ~= minGamma - 1
  display(sprintf('Unexpected: %d -> %d', gamma, minGamma));
end

display(sprintf('Obj Fun: %f, Constr Fun: %f, Gamma: %d', minObjFun, ...
		ConstrFun(UqSizes, minGamma), minGamma));

gamma = minGamma;


function [obj] = ObjFun(data, thres)

indices = find(data <= thres);
n = size(data, 1);

obj = sum(thres - data(indices)) / n;


function [constr] = ConstrFun(data, thres)

indices = find(data > thres);
n = size(data, 1);

constr = size(indices, 1) / n;

