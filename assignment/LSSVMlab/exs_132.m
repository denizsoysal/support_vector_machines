%% 1.3.2 Radnom split
clc, clf, clear
load iris.mat
gamlist = [0.001,0.01,0.1,1,10,100,1000];
sig2 = [0.001,0.01,0.1,1,10];


perfs_leaveoneout = [];
perfs_cross = [];
perfs_random = [];

gammas = [];
sigmas = [];




for sigma = sig2
    for gam =gamlist
        perf_leaveoneout = leaveoneout({ Xtrain , Ytrain , 'c', gam , sigma ,'RBF_kernel'}, 'misclass');
        perf_cross = crossvalidate({ Xtrain , Ytrain , 'c', gam , sigma ,'RBF_kernel'}, 10,'misclass');
        perf_random = rsplitvalidate({ Xtrain , Ytrain , 'c', gam , sigma ,'RBF_kernel'}, 0.80,'misclass');
        perfs_leaveoneout(end+1) = perf_leaveoneout;
        perfs_cross(end+1) = perf_cross;
        perfs_random(end+1) = perf_random;
        gammas(end+1) = gam;
        sigmas(end+1) = sigma;
    end
end

