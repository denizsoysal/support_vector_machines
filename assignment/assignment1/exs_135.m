load iris.mat;

gam =  0.1;  
sig2 = 1;

bay_modoutClass ({ Xtrain , Ytrain , 'c', gam , sig2 }, 'figure')