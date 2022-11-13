% In order to make an LS-SVM model, we need 2 extra parameters: gamma
% (gam) is the regularization parameter, determining the trade-off
% between the fitting error minimization and smoothness of the
% estimated function. sigma^2 (sig2) is the kernel function
% parameter of the RBF kernel:

%example of demofun : 

% gam = 10;
% sig2 = 0.3;
% type = 'function estimation';
% [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

%To evaluate new points for this model, the function simlssvm is used. 

%Yt = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xt);

%then plot : 

%plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear,clc,clf
close all;

%dataset : noisy sinc
X = ( -3:0.01:3)';
Y = sinc (X) + 0.1.* randn ( length (X), 1);

%training and test sets :

Xtrain = X (1:2: end);
Ytrain = Y (1:2: end);
Xtest = X (2:2: end);
Ytest = Y (2:2: end);


gam = 10.271 ; 
sig2 = 0.153834;
type = 'function estimation';

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
Yt = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
hold on
err = immse(Yt,Ytest)
plot(Xtest, Ytest,'b.')
plot(Xtest,Yt,'r-.'); 
xlabel('x')
ylabel('y')
title(['gamma=', num2str(gam),',  ' ,'sig2 =', num2str(sig2), ',  ', 'MSE on test=',num2str(err)])


%parameter tunning with tunelssvm and gridsearch/simplex

%automatic parameter tuning 


% The tuning procedure is fully automized in the LS-SVMlab toolbox. Type:
% >> [gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],
% 'RBF_kernel'}, 'algorithm', 'crossvalidatelssvm',{10, 'misclass'});
% where 'algorithm' can be chosen as 'simplex' (Nelder-Mead method) or 'gridsearch'
% (brute force gridsearch).

%we use MSE as cost function

load iris.mat;

%grid search
[gam_grid,sig2_grid , cost_grid ] = tunelssvm ({ Xtrain , Ytrain , 'f', [], [],'RBF_kernel'}, 'gridsearch', 'leaveoneoutlssvm',{'mse'});
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam_grid,sig2_grid,'RBF_kernel'});
Yt = simlssvm({Xtrain,Ytrain,type,gam_grid,sig2_grid,'RBF_kernel','preprocess'},{alpha,b},Xtest);
cost_grid

%simplex
[gam_simplex,sig2_simplex , cost_simplex ] = tunelssvm ({ Xtrain , Ytrain , 'f', [], [],'RBF_kernel'}, 'simplex', 'leaveoneoutlssvm',{'mse'});
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam_simplex,sig2_simplex,'RBF_kernel'});
Yt = simlssvm({Xtrain,Ytrain,type,gam_simplex,sig2_simplex,'RBF_kernel','preprocess'},{alpha,b},Xtest);
cost_simplex



