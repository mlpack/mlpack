% In this function, we solve the non-convex optimization problem
% encountered in the Maximum Margin Clustering via CCCP
%
% Input:
%   omega_0:    initial value for omega
%   b_0:        initial value for b
%   xi_0:       initial value for xi
%   C:          regularization parameter in the objective function, soft
%               constraint
%   W:          set of constraints, matrix
%   l:          bound on the label bias
%   MyData:     data set, each sample corresponds to a vector in MyData
%
% Call:     [omega,b,xi] = CCCP_MMC_primal(omega_0,b_0,xi_0,C,epsilon,W,l,MyData)
%
% Author: Bin Zhao
% Created on: 10 July, 2007
% Last updated on : 13 Nov, 2007

function [omega,b,xi] = CCCP_MMC_primal(omega_0,b_0,xi_0,C,W,l,MyData)

% Step 1: initialzation
[nConstraint nData] = size(W);
[nDim temp] = size(omega_0);
omegaOld = omega_0;
bOld = b_0;
xiOld = xi_0;
fValOld = 0.5 * omegaOld' * omegaOld + C * xiOld;

XInit = [omega_0; b_0; xi_0];

flagQuit = 0;
perQuit = 0.01; % perQuit controls the total step of the CCCP iteration

iter = 0;

HQP = [diag(ones(nDim, 1)) zeros(nDim, 2); zeros(2, nDim) ...
       zeros(2)];
fQP = zeros(nDim + 2, 1);
fQP(nDim + 2, 1) = C;

c_k = mean(W,2);
bQP = [0; l; l; -c_k];

x_k = sum(MyData,2);
beta_t = zeros(nData, 1);

s_k = zeros(nConstraint,1);
z_k = zeros(nDim,nConstraint);

% XOld = rand(nConstraint + 2,1);

% Step 2: CCCP iteration
% We solve the QP problem in its primal form
% display(sprintf('CCCP: %d : %f, %f', iter, fValOld));
minFVal = Inf;
XOld = XInit;
while (flagQuit == 0)
    iter = iter + 1;
    
    temp_z_k = zeros(nDim,nData);
    temp_s_k = zeros(nData,1);
    for iData = 1:nData
        temp_s_k(iData) = sign(omegaOld'*MyData(:,iData)+bOld);
        temp_z_k(:,iData) = temp_s_k(iData) * MyData(:,iData);
    end;
    
    s_k = W * temp_s_k / nData;
    z_k = temp_z_k * W' / nData;
    
    
    AQP = [zeros(1,nDim+1) -1;...
	  -x_k' -nData 0;...
	  x_k' nData 0;...
	  -z_k' s_k -ones(nConstraint, 1)];

    LB = [-Inf*ones(nDim+2,1)];
    LB(nDim+2) = 0;
    UB = [Inf*ones(1,nDim+2)]';
    % UB(nConstraint+1, 1) = Inf;
    % UB(nConstraint+2, 1) = Inf;
    
    ops = optimset('LargeScale', 'off', 'Display', 'off'); %, 'Diagnostics', ...
		   %'off');
    [XQP,fVal,exitFlag, output, lambda] = quadprog(HQP,fQP,AQP,bQP,...
						   [],[],LB,UB,[],ops);
    % display(fVal);
    if (exitFlag <= 0)
      display(exitFlag);
    end
    
    omegaOld = XQP(1:nDim);
    %xiOld = (-fVal - 0.5 * omegaOld' * omegaOld) / C;
    xiOld = XQP(nDim+2);
    %SV_index = find(XQP(1:nConstraint)>0);
%    if numel(SV_index) == 0
%      display(sprintf('||omega||^2 = %f, xi = %f', omegaOld' * ...
%		      omegaOld, xiOld));
%      break;
%    end
    %bOld = (c_k(SV_index(1)) - xiOld - omegaOld' *
    %z_k(:,SV_index(1))) / s_k(SV_index(1));
    bOld = XQP(nDim + 1);
    
    fVal = 0.5 * omegaOld' * omegaOld + C * xiOld;
    
    if (abs(sum(omegaOld' * MyData + bOld)) > l + 0.0001) 
      display(sprintf('WTF:%d %f', exitFlag,...
		      abs(sum(omegaOld'*MyData + bOld))));
      %display(output);
      %display(lambda.ineqlin);
      %display(lambda.eqlin);
      %display(XQP(length(XQP)-1:length(XQP)));
    end
    

    if(fValOld - fVal >= 0 & fValOld - fVal < perQuit * fValOld)
        flagQuit = 1;
    else
      %if fVal < minFVal
	%minFVal = fVal;
	%minIter = iter;
	
	%minOmegaOld = omegaOld;
	%minXiOld = xiOld;
	%minBOld = bOld;
      %end
      
      %if mod(iter, 100) == 0
	%display(sprintf('CCCP: %d : %f - %f, %f - %d', ...
	%  		iter, fVal, fValOld, minFVal, minIter));
	
	% use the minimum value found up until 100 iterations of
        % CCCP
	%omegaOld = minOmegaOld;
	%xiOld = minXiOld;
	%bOld = minBOld;

	%fVal = 0.5 * omegaOld' * omegaOld + C * xiOld;
	
	%display(sprintf('CCCP: force halt min fVal = %f', fVal));
	%flagQuit = 1;
      %end
      
      fValOld = fVal;
      XOld = XQP;
    end;
end;

omega = omegaOld;
b = bOld;
xi = xiOld;
display(sprintf('CCCP complete'));
