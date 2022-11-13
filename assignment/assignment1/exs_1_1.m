clear all;
close all;

%generate 2 Gaussian distributed classes with the same covariance matrix 
% and classify them

%input data
X1 = randn (50 ,2) + 1;
X2 = randn (51 ,2) - 1;
%corresponding labels
Y1 = ones (50 ,1);
Y2 = -ones (51 ,1);

%visualize data
figure ;
hold on;
%class 1 : 
plot (X1 (: ,1) , X1 (: ,2) , 'ro',LineWidth=10);
%class 2 :
plot (X2 (: ,1) , X2 (: ,2) , 'bo',LineWidth=10);
%linear classifier :
x = linspace(-4, 4,100);
y = linspace(4, -4, 100);
plot(x, y,linewidth=6)


title('Distribution of the data', fontsize=28);
legend('Class 1', 'Class -1', fontsize=20);
hold off;