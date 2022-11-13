% In addition to parameter tuning, the Bayesian framework can also be used to select the
% most relevant inputs by Automatic Relevance Determination (ARD). The following proce-
% dure uses this criterion for backward selection for a three dimensional input selection task,
% constructed as (use tuned gam and sig2 parameter values)

% For a given problem, one can determine the most relevant inputs
%   for the LS-SVM within the Bayesian evidence framework. To do so,
%   one assigns a different weighting parameter to each dimension in
%   the kernel and optimizes this using the third level of
%   inference. According to the used kernel, one can remove inputs
%   corresponding the larger or smaller kernel parameters. This
%   routine only works with the 'RBF_kernel' with a sig2 per
%   input. In each step, the input with the largest optimal sig2 is
%   removed (backward selection). For every step, the generalization
%   performance is approximated by the cost associated with the third
%   level of Bayesian inference
clear all;
clear,clc,clf
close all;
% 
% %dataset : noisy sinc
% X = ( -3:0.01:3)';
% Y = sinc (X) + 0.1.* randn ( length (X), 1);
% 
% %training and test sets :
% 
% Xtrain = X (1:2: end);
% Ytrain = Y (1:2: end);
% Xtest = X (2:2: end);
% Ytest = Y (2:2: end);

gam = 1.3587;
sig2 = 0.21785;

X = 6.* rand (100 , 3) - 3;
Y = sinc (X(: ,1)) + 0.1.* randn (100 ,1) ;
[ selected , ranking ] = bay_lssvmARD ({X, Y, 'f', gam , sig2 });

%figure of ranking

bar(ranking)
xlabel('Inputs') 
ylabel('Relevance Ranking') 


%regression task with each of input 

%let's divide onto training and test
Xtrain = X (1:80,:);
Ytrain = Y (1:80,:);
Xtest = X (81:end,:);
Ytest = Y (81:end,:);


%input 1 : 

type = 'function estimation';
[alpha,b] = trainlssvm({Xtrain(:,1),Ytrain,type,gam,sig2,'RBF_kernel'});
Yt = simlssvm({Xtrain(:,1),Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest(:,1));
figure
hold on
err = immse(Yt,Ytest)
plot(Xtest(:,1), Ytest,'b.')
plot(Xtest(:,1),Yt,'r.'); 
xlabel('x')
ylabel('y')
title(['gamma=', num2str(gam),',  ' ,'sig2 =', num2str(sig2), ',  ', 'MSE on test=',num2str(err)])
hold off


%input 2 : 


type = 'function estimation';
[alpha,b] = trainlssvm({Xtrain(:,2),Ytrain,type,gam,sig2,'RBF_kernel'});
Yt = simlssvm({Xtrain(:,2),Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest(:,2));
figure
hold on
err = immse(Yt,Ytest)
plot(Xtest(:,2), Ytest,'b.')
plot(Xtest(:,2),Yt,'r.'); 
xlabel('x')
ylabel('y')
title(['gamma=', num2str(gam),',  ' ,'sig2 =', num2str(sig2), ',  ', 'MSE on test=',num2str(err)])
hold off



%input 3 :

type = 'function estimation';
[alpha,b] = trainlssvm({Xtrain(:,3),Ytrain,type,gam,sig2,'RBF_kernel'});
Yt = simlssvm({Xtrain(:,3),Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest(:,3));
figure
hold on
err = immse(Yt,Ytest)
plot(Xtest(:,3), Ytest,'b.')
plot(Xtest(:,3),Yt,'r.'); 
xlabel('x')
ylabel('y')
title(['gamma=', num2str(gam),',  ' ,'sig2 =', num2str(sig2), ',  ', 'MSE on test=',num2str(err)])
hold off




