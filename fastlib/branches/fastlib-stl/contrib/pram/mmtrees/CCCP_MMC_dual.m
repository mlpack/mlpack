function [omega,b,xi,fail] = CCCP_MMC_dual(omega_0,b_0,xi_0,C,W,l,MyData)
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
[nConstraint, nData] = size(W);
[nDim temp] = size(omega_0);
omegaOld = omega_0;
bOld = b_0;
xiOld = xi_0;
fValOld = 0.5 * omegaOld' * omegaOld + C * xiOld;

first_loop = 1;
floop = 1;

flagQuit = 0;
perQuit = 0.01; % perQuit controls the total step of the CCCP iteration

iter = 0;
c_k = mean(W,2);
s_k = zeros(nConstraint,1);
z_k = zeros(nDim,nConstraint);
x_k = sum(MyData,2);

if (isnan(sum(x_k)))
  display(sprintf('Data NaN: %f', sum(x_k)));
  keyboard;
end

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

    if (isnan(bOld))
      display(sprintf('B old: %f', bOld));
      keyboard;
    end
    
    temp_z_k = zeros(nDim,nData);
    temp_s_k = zeros(nData,1);
    for iData = 1:nData
        temp_s_k(iData) = sign(omegaOld'*MyData(:,iData)+bOld);
        temp_z_k(:,iData) = temp_s_k(iData) * MyData(:,iData);
    end;
    
    s_k = W * temp_s_k / nData;
    z_k = temp_z_k * W' / nData;
    
    Tmp = sum(sum(z_k, 1));
    if isnan(Tmp)
      display(sprintf('Z_k NaNed: %f', Tmp));
      keyboard;
    end
        
    x_mat = [z_k,-x_k,x_k];
    HQP = x_mat' * x_mat;
    fQP = [-c_k;l;l];
    
    AQP = [ones(1,nConstraint),0,0];
    bQP = C;
    
    Aeq = [-s_k',nData,-nData];
    beq = 0;
    
    LB = zeros(nConstraint+2,1);
    UB = [Inf*ones(1,nConstraint+2)]';
    % UB(nConstraint+1, 1) = Inf;
    % UB(nConstraint+2, 1) = Inf;
    
    
    %Tmp = sum(sum(HQP, 1));
    %display(sprintf('*:%f',Tmp));
    %if (isnan(Tmp))
    %  display(Tmp);
    %  keyboard;      
    %end
    
    
    %if (rank(HQP) < 1) 
    %  display(sprintf('rH: %f', rank(HQP)));
    %  fail = 1;
    %else
    
    
    ops = optimset('LargeScale', 'off', 'Display', 'off');
    [XQP,fVal,exitFlag, output, lambda] = quadprog(HQP,fQP,AQP,bQP,...
						   Aeq,beq,LB,UB, ...
						   [],ops);
    
    %display(fVal);
    if (exitFlag <= 0)
      %display(exitFlag);
      %keyboard;
      fail = 1;
    else
      fail = 0;
    end
    
    
    %display(sum(XQP.*XQP));
    omegaTemp = omegaOld;
    omegaOld = x_mat * XQP;
    xiOld = (-fVal - 0.5 * omegaOld' * omegaOld) / C;
    SV_index = find(XQP(1:nConstraint)>0);
    
    % in case no constraints were > 0 (meaning not the theoretical
    % dual optimal, not even close) b cannot be computed from the
    % KKT conditions. Hence we would have to break the CCCP routine
    % and return a failure
    ind_b = -1;
    if numel(SV_index) > 0
      for i = 1:numel(SV_index)
	if s_k(SV_index(i)) ~= 0
	  ind_b = i;
	  break;
	end
      end
    end

    if ind_b == -1 
      
      %keyboard;
      
      if XQP(nConstraint+1) > 0
	% \sum <w, x_i> + b == -l
	bOld = (-l - omegaOld' * x_k) / nData;
      else
	if XQP(nConstraint+2) > 0
	  bOld = (l - omegaOld' * x_k) / nData;
	else
	  fprintf(1, ['no tight constraints, hence no way to compute' ...
		      ' b\n']);
	  b_UB = (l - omegaOld' * x_k) / nData;
	  b_LB = (-l - omegaOld' * x_k) / nData;
	  
	  bOld = (b_UB + b_LB) / 2;
	  %keyboard;
	  
	end
      end
    
    else
     
      bOld = (c_k(SV_index(ind_b)) - xiOld - ...
	      omegaOld' * z_k(:,SV_index(ind_b)))...
	     / s_k(SV_index(ind_b));
      
      if isnan(bOld)
	display(sprintf('B new: %f', bOld));
	keyboard;
      end
      
    end
    
            
    fVal = 0.5 * omegaOld' * omegaOld + C * xiOld;
    
    % if the solution is feasible with the balance constraint
    if (abs(sum(omegaOld' * MyData + bOld)) <= l + 0.0001)

      if floop == 1
	%fprintf(1, '');
	floop = 0;
      end
      
      % check if you are ready to stop
      if (fValOld - fVal) >= 0 &...
	   fValOld - fVal < perQuit * fValOld
	flagQuit = 1;
	fail = 0;
	  %display(fValOld); 
	  %display(fVal);
      %else
	% update the old values and move on
	%if (fValOld >= fVal)
	%end
      end;

      if fValOld - fVal > 0 | first_loop == 1
	omega = omegaOld;
	b = bOld;
	xi = xiOld;
	fValOld = fVal;
	first_loop = 0;
      end
      
    end
    if iter > 100 | fail == 1
      fail = 0;
      %display(fail);
      %keyboard;
      flagQuit = 1;
      if floop == 1 % never updated
	omega = omega_0;
	b = b_0;
	xi = xi_0;
	fail = 1;
      end
      
    end
end;

%omega = omegaOld;
%b = bOld;
%xi = xiOld;
if (abs(sum(omega' * MyData + b)) > l + 0.0001) 
  display(sprintf('WTF O:%d %f %f %d', exitFlag,...
		  abs(sum(omega'*MyData + b)), l, iter));
end
%display(sprintf('CCCP complete'));
