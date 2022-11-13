clc, clf, clear

%automatic parameter tuning 


% The tuning procedure is fully automized in the LS-SVMlab toolbox. Type:
% >> [gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],
% 'RBF_kernel'}, 'algorithm', 'crossvalidatelssvm',{10, 'misclass'});
% where 'algorithm' can be chosen as 'simplex' (Nelder-Mead method) or 'gridsearch'
% (brute force gridsearch).

load iris.mat;
[gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});
