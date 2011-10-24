function tf = topic_based_censoring(tf, D, k, p)
%function tf = topic_based_censoring(tf, D, k, p)
%
% tf is the data
% k is the topic to censor (the top 10 words of the topic will be censored)
% p is the proportion of documents to censor
%
% Note: We only censor the words in a topic with the largest
% POSITIVE weights. Negative weights appear to be encountered only
% rarely, and they also are less interpretable so it's unclear if
% the absolute value of the weights should be considered when
% identifying the top 10 words.

[y i] = sort(D(:,k));

word_inds = i(end-9:end)';

for word_ind = word_inds
  fprintf('%d.\n', word_ind);
  doc_inds = find(tf(word_ind,:));
  n_docs_with_word = length(doc_inds);

  n_docs_to_censor = round(p * n_docs_with_word)

  tmp = randperm(n_docs_with_word);
  doc_inds_to_censor = doc_inds(tmp(1:n_docs_to_censor));
  %disp(doc_inds_to_censor');
  tf(word_ind,doc_inds_to_censor) = 0;
end
