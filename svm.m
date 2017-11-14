% normalizesvm
function result = normalizekm(km)
% Copyright 2012 Nino Shervashidze, Karsten Borgwardt
% normalizes kernelmatrix km
% such that diag(result) = 1, i.e. K(x,y) / sqrt(K(x,x) * K(y,y))
% @author Karsten Borgwardt
% @date June 3rd 2005
% all rights reserved

nv = sqrt(diag(km));
nm =  nv * nv';
knm = nm .^ -1;
for i = 1:size(knm,1)
for j = 1:size(knm,2)
if (knm(i,j) == Inf)
knm(i,j) = 0;
end
end
end
result = km .* knm;

% run_independent

function [result,mean_accuracy,std_accuracy] = runIndependent(K,lk)
% Copyright 2012 Nino Shervashidze, Karsten Borgwardt
% K = kernel matrix (n*n)
% lk = vector of labels (n*1)
% cv = number of folds in cross-validation

% standard deviation
% independent scheme
% best c

addpath('~/code/libsvm');
% randomly permute kernel matrix and labels
r = randperm(size(K,1));
K = K(r,r);
lk = lk(r);
lk=lk';
lkoriginal = lk;
Koriginal = K;

%% stratified cross-validation
%sum(sum(Koriginal));
%neworder = stratifiedsplit(lk)
%for i = 1:size(neworder,2)
%m = size(neworder(i).old,1);
%r = randperm(m);
%newlk(neworder(i).new) =  lk(neworder(i).old(r));
%Knew([neworder(i).new]',[neworder(i).new]') = K([neworder(i).old(r)]',[neworder(i).old(r)]');
%end
%
%sum(sum(Knew)) - sum(sum(Koriginal))
%dbstop
%lk = newlk'
%K = Knew;
%size(lk);
%size(K);
%dbstop
%Koriginal = K;


% bring kernel matrix into libsvm format
p80 = ceil(size(K,2) * 0.8);
p90 = ceil(size(K,2) * 0.9);

% specify range of c-values
cvalues = (10 .^ [-7:2:7]) / size(K,2);

cv = 10;
fs = size(K,2) - p90;

% cross-validation loop
for k = 1:cv
K = Koriginal;
lk = lkoriginal;

K = K([k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs],[k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs]);
lk = lk([k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs]);
K = makepos(K);
K1 = [(1:size(K,1))', normalizekm(K)];
      
      %if any(strcmp('optimal',options))
      imresult=[];
      for i = 1:size(cvalues,2)
      % train on 80%, predict on 10% (from 81% to 90%)
      size(lk(1:p80));
      size(K1(1:p80,1:p80+1));
      model = svmtrain(lk(1:p80,1), K1(1:p80,1:p80+1), strcat(['-t 4  -c ' num2str(cvalues(i))]));
      [predict_label, accuracy, dec_values] = svmpredict(lk(p80+1:p90,1),K1(p80+1:p90,1:p80+1), model);
      accuracy80 = accuracy;
      imresult(i)= accuracy(1);
      end
      
      
      % determine optimal c
      [junk,optimalc]= max(fliplr(imresult));
      optimalc = size(cvalues,2)+1 - optimalc;
      % train on 90% with optimal c, predict on 10% (from 91% to 100%)
      model = svmtrain(lk(1:p90,1), K1(1:p90,1:p90+1),strcat(['-t 4  -c ' num2str(cvalues(optimalc))]) );
      [predict_label, accuracy, dec_values] = svmpredict(lk(p90+1:size(K,1),1), K1(p90+1:size(K,1),1:p90+1), model);
      accuracy90 = accuracy
      result(k)=accuracy(1)
      
      
      end
      mean_accuracy =  mean(result)
      std_accuracy = std(result)
      
      
      end
      
      %
      %% cross-validation
      %if any(strcmp('cv',options))
      %options = strcat(['-t 4 -v ' num2str(cv) ' -c ' num2str(cvalues(i))])
      %result(i) = svmtrain(lk, K1, options); %', num2str(cv)));
      %end
      %
      %end
      
      function result = makepos(K)
      pd = 0;
      addc = 10e-7;
      while (pd ==  0)
      
      addc = addc * 10
      try
      if (isinf(addc) == 1)
      pd = 1;
      else 
      chol(normalizekm(K + eye(size(K,1),size(K,1)) * addc));
      pd = 1;
      end
      catch
      
      end
      
      end
      if (isinf(addc)==0)
      result = K + eye(size(K,1),size(K,1)) * addc;
      else
      result = eye(size(K,1));
      end
      end

% run_multipli

      function res = runmultiplesvm(Ks,lk,n)
      % Copyright 2012 Nino Shervashidze, Karsten Borgwardt
      % Input: Ks - cell array of h  m x m kernel matrices
      %        lk - m x 1 array of class labels
      %        n - number of times we want to run svm
      % Output: res is a 1 x n+1 array of structures. Each of the first n
      %         elements contains fields optkernel, optc, accuracy, mean_acc and std_acc,
      %         and res(n+1) has only two fields - the mean and the std of the n
      %         mean_acc's
      %
      s=0;
      for i=1:n
      res(i)=runsvm(Ks,lk);
      meanacc(i) = res(i).mean_acc;
      end
      res(n+1).mean_acc = mean(meanacc);
      res(n+1).std_acc = std(meanacc);
      end
      
      
%svm
      function [res] = runsvm(Ks,lk)
      % Copyright 2012 Nino Shervashidze, Karsten Borgwardt
      % runsvm(Ks,lk)
      % K = 1 x h cell array of kernelmatrices (n*n)
      % lk = vector of class labels (n*1)
      % cv = number of folds in cross-validation
      
      % independent scheme
      % best c
      
      addpath('~/code/libsvm');
      n=length(lk) % size of the dataset
      % randomly permute labels: r will be also used for permuting the kernel matrices
      r = randperm(n);
      lk = lk(r);
      
      % specify range of c-values
      cvalues = (10 .^ [-7:2:7]) / size(lk,1);
      
      cv = 10;
      p80 = ceil(n * (1-2/cv));
      p90 = ceil(n * (1-1/cv));
      fs = n - p90; % fold size
      
      
      % output variables
      res.optkernel=zeros(cv,1);
      res.optc=zeros(cv,1);
      res.accuracy=zeros(cv,1);
      
      % cross-validation loop
      opth=zeros(1,cv);
      for k = 1:cv
      imresult=[];
      
      height = length(Ks);
      for h=1:height
      K = Ks{h}(r,r);
      K_current = K([k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs],[k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs]);
      lk_current = lk([k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs]);
      K_current = makepos(K_current);
      K1 = [(1:size(K_current,1))', normalizekm(K_current)];
            
            
            for i = 1:size(cvalues,2)
            % train on 80%, predict on 10% (from 81% to 90%)
            size(lk_current(1:p80));
            size(K1(1:p80,1:p80+1));
            model = svmtrain(lk_current(1:p80,1), K1(1:p80,1:p80+1), strcat(['-t 4  -c ' num2str(cvalues(i))]));
            [predict_label, accuracy, dec_values] = svmpredict(lk_current(p80+1:p90,1),K1(p80+1:p90,1:p80+1), model);
            imresult(h,i)= accuracy(1);
            end
            end
            
            % determine optimal h and c
            [junk,position]= max(imresult(:));
            [optimalh, indoptimalc]=ind2sub(size(imresult),position);
            
            opth(k)=optimalh;
            res.optc(k)= cvalues(indoptimalc);
            res.optkernel(k)=optimalh;
            % train on 90% with optimal c, predict on 10% (from 91% to 100%)
            K = Ks{optimalh}(r,r);
            K_current = K([k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs],[k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs]);
            lk_current = lk([k*fs+1:size(K,2),1:(k-1)*fs,(k-1)*fs+1:k*fs]);
            K_current = makepos(K_current);
            K1 = [(1:size(K_current,1))', normalizekm(K_current)];
                  
                  model = svmtrain(lk_current(1:p90,1), K1(1:p90,1:p90+1),strcat(['-t 4  -c ' num2str(cvalues(indoptimalc))]) );
                  [predict_label, accuracy, dec_values] = svmpredict(lk_current(p90+1:size(K,1),1), K1(p90+1:size(K,1),1:p90+1), model);
                  res.accuracy(k)=accuracy(1)
                  end
                  res.mean_acc =  mean(res.accuracy) 
                  res.std_acc = std(res.accuracy)
                  end
                  
                  
                  function result = makepos(K)
                  pd = 0;
                  addc = 10e-7;
                  while (pd ==  0)
                  
                  addc = addc * 10
                  try
                  if (isinf(addc) == 1)
                  pd = 1;
                  else 
                  chol(normalizekm(K + eye(size(K,1),size(K,1)) * addc));
                  pd = 1;
                  end
                  catch
                  
                  end
                  
                  end
                  if (isinf(addc)==0)
                  result = K + eye(size(K,1),size(K,1)) * addc;
                  else
                  result = eye(size(K,1));
                  end
                  end

% result
      function result = runntimes(K,lk,n)
      % Copyright 2012 Nino Shervashidze, Karsten Borgwardt
      % Input: K  - m x m kernel matrix
      %        lk - m x 1 array of class labels
      %        n - number of times we want to run svm
      % Output: result - a structure with fields accuracy, mean, std and
      %                  mean, std and se are the mean, the standard
      %                  deviation and the standard error of accuracy
      
      accuracy = zeros(n,1);
      
      for i = 1:n
      [junk1, accuracy(i), junk2] = runIndependent(K, lk)
      
      end
      
      result.mean = mean(accuracy)
      
      result.std = std(accuracy)
      
      result.se = result.std / sqrt(n)
      
      result.accuracy = accuracy
      
      result
