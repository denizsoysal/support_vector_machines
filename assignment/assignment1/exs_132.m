%% 1.3.2 Radnom split
clc, clf, clear
load iris.mat
gamlist = [0.001,0.01,0.1,1,10];
sig2 = [0.001,0.01,0.1,1,10];


perfs = [];
gammas = [];
sigmas = [];




for sigma = sig2
    for gam =gamlist
        perf_leaveoneout = leaveoneout({ Xtrain , Ytrain , 'c', gam , sigma ,'RBF_kernel'}, 'misclass');
        perf_cross = crossvalidate({ Xtrain , Ytrain , 'c', gam , sigma ,'RBF_kernel'}, 10,'misclass');
        perf_random = rsplitvalidate({ Xtrain , Ytrain , 'c', gam , sigma ,'RBF_kernel'}, 0.80,'misclass');
        gam,sigma,perf
        perfs(end+1) = perf;
        gammas(end+1) = gam;
        sigmas(end+1) = sigma;
    end
end

results = table(gammas,sigmas,perfs)

