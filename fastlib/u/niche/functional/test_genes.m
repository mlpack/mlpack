unknown_phase = -1;
M_G1_boundary_phase = 0;
G1_phase = 1;
S_phase = 2;
S_G2_phase = 3;
G2_M_phase = 4;

unknown_cluster = -1;
CLB2_cluster = 0;
CLN1_cluster = 1;
Histone_cluster = 2;
MAT_cluster = 3;
MCM_cluster = 4;
MET_cluster = 5;
SIC1_cluster = 6;
Y_cluster = 7;


t = 0:7:119;

data = zeros(82, 6178);
phases = zeros(1,6178);
clusters = zeros(1,6178);

fid = fopen('genes/combined_phase_cluster.txt');

% throw away first line
textscan(fid, '%s', 83, 'delimiter', '\t', 'emptyValue', -inf);

cur_row = [1];

row_num = 0;

while(length(cur_row) > 0)
  %  phasestring %
  phasestring = textscan(fid, '%s', 1, 'delimiter', '\t', 'emptyValue', ...
			 -inf);
  phasestring = char(phasestring{1});
  if strcmp(phasestring, 'm_g1_boundary') == 1
    phase = M_G1_boundary_phase;
  elseif strcmp(phasestring, 'g1') == 1
    phase = G1_phase;
  elseif strcmp(phasestring, 's') == 1
    phase = S_phase;
  elseif strcmp(phasestring, 's_g2') == 1
    phase = S_G2_phase;
  elseif strcmp(phasestring, 'g2_m') == 1
    phase = G2_M_phase;
  elseif strcmp(phasestring, 'unknown') == 1
    phase = unknown_phase;
  else
    disp(phasestring);
  end
  
  %  clusterstring %
  clusterstring = textscan(fid, '%s', 1, 'delimiter', '\t', 'emptyValue', ...
			   -inf);
  clusterstring = char(clusterstring{1});
  if strcmp(clusterstring, 'CLB2') == 1
    cluster = CLB2_cluster;
  elseif strcmp(clusterstring, 'CLN2') == 1
    cluster = CLN1_cluster;
  elseif strcmp(clusterstring, 'Histone') == 1
    cluster = Histone_cluster;
  elseif strcmp(clusterstring, 'MAT') == 1
    cluster = MAT_cluster;
  elseif strcmp(clusterstring, 'MCM') == 1
    cluster = MCM_cluster;
  elseif strcmp(clusterstring, 'MET') == 1
    cluster = MET_cluster;
  elseif strcmp(clusterstring, 'SIC1') == 1
    cluster = SIC1_cluster;
  elseif strcmp(clusterstring, 'Y') == 1
    cluster = Y_cluster;
  elseif strcmp(clusterstring, 'unknown') == 1
    cluster = unknown_cluster;
  else
    disp(clusterstring);
  end
  
  
  textscan(fid, '%s', 1, 'delimiter', '\t', 'emptyValue', -inf);
  cur_row = textscan(fid, '%f', 82, 'delimiter', '\t', ...
		     'emptyValue', -inf);
  cur_row = cur_row{1};
  
  if(length(cur_row) > 0)
    row_num = row_num + 1;
    while length(cur_row) < 82
      cur_row(end+1) = -inf;
    end
    data(:,row_num) = cur_row;
    phases(row_num) = phase;
    clusters(row_num) = cluster;
  end
  
  %fprintf('%f\n', length(cur_row));
end

good_indices = [];
for i = 1:size(data,2)
  if length(find(data(7:7+17,i) == -inf)) <= 1
    good_indices(end+1) = i;
  end
end



data = data(7:7+17, good_indices);
phases = phases(good_indices);
clusters = clusters(good_indices);

% indicate missing values with NaN, as required by the fd tools data2fd()
data(find(data == -inf)) = NaN;

% data ready!


N = size(data,2);
p = 17;

mybasis = create_bspline_basis([min(t) max(t)], p, 4);
basis_curves = eval_basis(0:1:119, mybasis);
basis_inner_products = full(eval_penalty(mybasis, int2Lfd(0)));

myfd_data = data2fd(data, t, mybasis);

lambda = 0;
myfdPar = fdPar(mybasis, 2, lambda);

fprintf('calling funcica\n');
[ic_curves, ic_coef, Y, pc_coef, pc_curves, pc_scores, W] = ...
    funcica(t, myfd_data, p, basis_curves, myfdPar, basis_inner_products);
fprintf('funcica returned\n');

p_small = size(ic_coef, 2);

pc_curves = basis_curves * pc_coef;

ic_curves = pc_curves * W';

pc_coef = pc_coef';



% rescale the utilized parts of pc_coef such that
% the pc_curves square integrate to 1

for j = 1:p_small
  pc_coef_j = pc_coef(j,:);
  alpha = sqrt(sum(sum((pc_coef_j' * pc_coef_j) .* basis_inner_products)));
  pc_coef(j,:) = pc_coef(j,:) / alpha;
end

pc_scores = pc_scores';

ic_scores = W * pc_scores;


magnitude = sum(sum(pc_scores .^ 2));


for i = 1:p_small
  h_pc(i) = get_vasicek_entropy_estimate_std(pc_scores(i,:));
  h_ic(i) = get_vasicek_entropy_estimate_std(ic_scores(i,:));
end

h_pc_sum = sum(h_pc);
h_ic_sum = sum(h_ic);





% discriminant analysis using pc features %
% pc_scores is d x N

g1_indices = find(phases == G1_phase);
nong1_indices = find(phases ~= G1_phase);

svm_data = [pc_scores(:,g1_indices) pc_scores(:,nong1_indices)]';
svm_labels = [1 * ones(length(g1_indices),1);
	      -1 * ones(length(nong1_indices),1)];


latestSVM = svml('latestSVM','Kernel',1,'KernelParam',3,'C',1, ...
		 'ExecPath','/home/niche/matlab/toolboxes/svml');

for i=1:size(svm_data, 1)
  latestSVM = ...
      svmltrain(latestSVM, [svm_data([1:(i-1) (i+1):end], :)], ...
		svm_labels([1:(i-1) (i+1):end]));
  ypred(i) = svmlfwd(latestSVM, svm_data(i,:), svm_labels(i));
end
