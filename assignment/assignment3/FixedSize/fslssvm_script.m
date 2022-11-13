%data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:3000,:);
%data = load('california.dat','-ascii'); function_type = 'f';
%use this for california to only consider first 2000 data points, because training on whole dataset takes too much time :
% data = data(1:2000,1:end)
% addpath('../LSSVMlab')

X = data(:,1:end-1);
Y = data(:,end);

% % binarize the labels for shuttle data (comment these lines for
% % california!)
Y(Y == 1) = 1;
Y(Y ~= 1) = -1;
%count occurence:
% [GC,GR] = groupcounts(Y) 

testX = [];
testY = [];

figure
plot(X(:,8),Y,'b*')
%%

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 6;
kernel_type = 'poly_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);