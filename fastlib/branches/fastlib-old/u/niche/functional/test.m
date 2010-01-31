% test script

c3 = 14;
c4 = 18;

mybasis = create_bspline_basis([0 .5], 30, 4);

load ~/brains/bci_comp_2003_dataSet_IV/sp1s_aa_1000Hz;



x_train = CAR(x_train);
ground_truth = y_train;

num_train = 216;
% num_train = 316;

y_train = squeeze(x_train(:,c3,1:num_train));
y_test = squeeze(x_train(:,c3,217:end));

y_train_mean = mean(y_train,2);

y_train = y_train - repmat(y_train_mean, [1 num_train]);
y_test = y_test - repmat(y_train_mean, [1 100]);

% y_shifted = x_train_c3_shifted;

% y is num samples (500) by num epochs (318)
argvals = .001:.001:.5;

myfd_train = data2fd(y_train, argvals, mybasis);
myfd_test = data2fd(y_test, argvals, mybasis);
%myfd_shifted = data2fd(y_shifted, argvals, mybasis);

%pca_results = pca_fd(myfd_train, 30);
pca_results = pca_fd(myfd_train, 30, fdPar(mybasis, 2, 0));

% treat left and right data differently


harmfd = struct(pca_results.harmfd);
score_train = pca_results.harmscr;
smyfd_train = struct(myfd_train);
smyfd_test = struct(myfd_test);


for i=1:100
  score_test(i,:) = smyfd_test.coef(:,i)' * inv(harmfd.coef');
end





lefts_train = find(ground_truth(1:num_train) == 0);
rights_train = find(ground_truth(1:num_train) == 1);

lefts_test = find(ground_truth((num_train+1):end) == 0);
rights_test = find(ground_truth((num_train+1):end) == 1);


pc = 2;

lmean = mean(score_train(lefts_train, pc));
rmean = mean(score_train(rights_train, pc));


left_correct = ...
    abs(score_test(lefts_test, pc) - lmean) < ...
    abs(score_test(lefts_test, pc) - rmean);

right_correct = ...
    abs(score_test(rights_test, pc) - rmean) < ...
    abs(score_test(rights_test, pc) - lmean);

[sum(left_correct) / length(left_correct)
 sum(right_correct) / length(right_correct)]
