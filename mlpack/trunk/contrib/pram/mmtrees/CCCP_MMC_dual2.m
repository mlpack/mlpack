function [omega,b,xi,fail] = CCCP_MMC_dual2(omega_0,b_0,xi_0,C,W,l,MyData)
% In this function, we solve the non-convex optimization problem
% encountered in the Maximum Margin Clustering via CCCP
%
% Input:
%   omega_0:    initial value for omega
%   b_0:        initial value for b
%   xi_0:       initial value for xi
%   C:          regularization parameter in the objective function,
%               soft constraint
%   W:          set of constraints, matrix
%   l:          bound on the label bias
%   MyData:     data set, each sample corresponds to a vector in MyData
%
% Output:
%   omega:   The final value of omega
%   b:       The final value of b
%   xi:      The final value of xi
%   fail:    Whether the CCCP failed (in which case omega = 0, b = 0
%
% Call:[omega,b,xi,fail] = CCCP_MMC_dual(omega_0,b_0,xi_0,C,W,l,MyData)
%
% Author: Bin Zhao
% Created on: 10 July, 2007
% Last updated on : 13 Nov, 2007
%
% Modified by: Parikshit Ram
% Modified on: 11 January, 2011


% Step 1: initialzation
[nConstraint nData] = size(W);
[nDim temp] = size(omega_0);
omegaOld = [omega_0; b_0]; % avoiding the use of b
% bOld = b_0;
xiOld = xi_0;
fValOld = 0.5 * omegaOld' * omegaOld + C * xiOld; % but the
                                                  % objective changes

flagQuit = 0;
perQuit = 0.01; % perQuit controls the total step of the CCCP iteration

iter = 0;
c_k = mean(W,2);
s_k = zeros(nConstraint,1);
z_k = zeros(nDim,nConstraint);

% changing the data x to (x,1) to avoid having to compute b
newData = [MyData; ones(1, nData)];
x_k = sum(newData,2);
% XOld = rand(nConstraint + 2,1);

% Step 2: CCCP iteration
% We solve the QP problem in its dual form, which will be much faster than
% directly solving the primal problem
% display(sprintf('CCCP: %d : %f, %f', iter, fValOld));
minFVal = Inf;

% Trial
% fValOld = Inf;
noUpdate = 1;
fValPrev = 0;
stuckAtVal = 0;

totalIter = 0;

while (flagQuit == 0)
    iter = iter + 1;
    
    temp_z_k = zeros(nDim+1,nData);
    temp_s_k = zeros(nData,1);
    for iData = 1:nData
        temp_s_k(iData) = sign(omegaOld'*newData(:,iData));
        temp_z_k(:,iData) = temp_s_k(iData) * newData(:,iData);
    end;
    
    s_k = W * temp_s_k / nData;
    z_k = temp_z_k * W' / nData;
    
    x_mat = [z_k,-x_k,x_k];
    HQP = x_mat' * x_mat;
    fQP = [-c_k;l;l];
    
    AQP = [ones(1,nConstraint),0,0];
    bQP = C;
    
    %Aeq = [-s_k',nData,-nData];
    %beq = 0;
    
    LB = zeros(nConstraint+2,1);
    UB = [Inf*ones(1,nConstraint+2)]';
    % UB(nConstraint+1, 1) = Inf;
    % UB(nConstraint+2, 1) = Inf;
    
    ops = optimset('LargeScale', 'off', 'Display', 'off', 'MaxIter', ...
		   500);
    [XQP,fVal,exitFlag, output, lambda] = quadprog(HQP,fQP,AQP,bQP,...
						   [],[],LB,UB, ...
						   [],ops);
    
    %display(fVal);
    if (exitFlag <= 0)
      %display(exitFlag);
      fail = 1;
    else
      fail = 0;
    end
    
    %display(sum(XQP.*XQP));
    omegaTemp = omegaOld;
    omegaOld = x_mat * XQP;
    xiOld = (-fVal - 0.5 * omegaOld' * omegaOld) / C;

    fVal = 0.5 * omegaOld' * omegaOld + C * xiOld;
    %display(fVal);
      
    %    if (abs(sum(omegaOld' * newData + bOld)) > l + 0.0001) 
    %display(sprintf('WTF:%d %d %f %g = %g - %g', exitFlag, iter,...
    %	    abs(sum(omegaOld'*newData)), ...
    %	    fValOld - fVal, fValOld, fVal));
    
    if (abs(sum(omegaOld' * newData)) <= l + 0.0001)
      if(fValOld - fVal >= 0 & fValOld - fVal < perQuit * fValOld)
	flagQuit = 1;
	fail = 0;
      else
	%clear fValOld;
	if (fValOld >= fVal | noUpdate == 1)
	  fValOld = fVal;
	  noUpdate = 0;
	end
	  
	if fVal == fValPrev
	  stuckAtVal = stuckAtVal + 1;
	else
	  fValPrev = fVal;
	  stuckAtVal = 0;
	end
	  
	if stuckAtVal > 5
	  fValOld = fVal;
	  stuckAtVal = 0;
	end
	  
	%if iter > 80
	%  totalIter = totalIter + iter;
	%  iter = 0;
	    
	%  display(omegaOld'*omegaOld);
	    
	  %reinitialize the parameters
	%  omegaOld = omegaTemp + 0.0000003 * ones(nDim+1,1); % initial value
	%  xiOld = 0.05;
	    
	%end
	      
	if iter > 100
	  flagQuit = 1;
	  omegaOld = zeros(nDim+1, 1);
	  xiOld = 0;
	  fail = 1;
	  %display(fail);
	end
	  
      end;
    else
	if iter > 100
	  flagQuit = 1;
	  omegaOld = zeros(nDim+1, 1);
	  xiOld = 0;
	  fail = 1;
	  %display(fail);
	end
    end; % feasible solution
end;


omega = omegaOld(1:nDim);
b = omegaOld(nDim+1);
xi = xiOld;
if (abs(sum(omegaOld' * newData)) > l + 0.0001) 
  display(sprintf('WTF O:%d %f %d', exitFlag,...
		  abs(sum(omegaOld'*newData)), iter));
end
%display(sprintf('CCCP complete'));
