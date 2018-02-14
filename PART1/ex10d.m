%% setup
clear all
close all
load('boston.mat')
X=boston(:,1:end-1);
y=boston(:,end);
%initiliazation of parameters
trials=20;
gamma=2.^(-40:-26);
sigma=2.^(7:0.5:13);
[g,s]=meshgrid(gamma,sigma);
g=g(:);
s=s(:);
len_sigma=length(s)';
%% data preallocation
mse_train=zeros(trials,1);
mse_test=zeros(trials,1);
g_choice=zeros(trials,1);
s_choice=zeros(trials,1);
index=zeros(trials,1);

for ii=1:trials
    [X_train, y_train, X_test,y_test]=splitData(X,y);
    %% Perform parameter selection process
    fold5Score=zeros(len_sigma,1);
    for jj=1:len_sigma
        fold5Score(jj)=kFoldScore(X_train,y_train,g(jj),s(jj),5);
    end
    [val,id]=min(fold5Score);
    index(ii)=id;
    gamma=g(id);
    sigma=s(id);
    
    %% Compute MSE and log other data
    g_choice(ii)=gamma;
    s_choice(ii)=sigma;
    K_tr=Kernel_mat2(X_train,X_train,sigma);
    alpha = kridgereg( K_tr,y_train,gamma );
    mse_train(ii) = dualcost( K_tr,y_train,alpha );
    
    K_te=Kernel_mat2(X_test,X_train,sigma);
    mse_test(ii) = dualcost( K_te,y_test,alpha );
end
%% Print out means

[mean(mse_train) sqrt(var(mse_train));mean(mse_test) sqrt(var(mse_test))]


%% Section that was used for figure generation
gamma=2.^(-40:-26);
sigma=2.^(7:0.5:13);
[g,s]=meshgrid(gamma,sigma);
Sc=reshape(fold5Score,13,15);
surf(log(g),log(s),log(Sc));
grid off
h=gcf;
a=gca;
h.Color=[1 1 1];
xlabel('log(\gamma)')
ylabel('log(\sigma)')
zlabel('log(Score)')
a.XLabel.FontSize=14;
a.YLabel.FontSize=14;
a.ZLabel.FontSize=14;% 
a.XLabel.FontWeight='bold';
a.YLabel.FontWeight='bold';
a.ZLabel.FontWeight='bold';
title('5-fold-cross-validation');
a.XAxis.FontWeight='bold';
a.YAxis.FontWeight='bold';
a.ZAxis.FontWeight='bold';
a.XAxis.FontSize=12;
a.YAxis.FontSize=12;
a.ZAxis.FontSize=12;
%% k-fold-cross-validation routine
%This function takes a training set and a pair of parameters. Then performs
%k-fold cross validation and outputs the mean score for this set of
%parameters.
function score=kFoldScore(X_tr,y_tr,gamma,sigma,k)
set_size=floor(size(X_tr,1)/k);%size of single fold
rem=mod(size(X_tr,1),set_size);%remainder sample size
ids=cell(k,1);%indexes for folds

for ii=1:k %prepare set indeces
    ids{ii}=set_size*(ii-1)+1:set_size*ii;
end

ids{k}=[ids{k} size(X_tr,1)-rem+1:size(X_tr,1)];
scores=zeros(k,1);
K_tr=Kernel_mat2(X_tr,X_tr,sigma);% compute K for all folds
for ii=1:k
    ev_y=y_tr(ids{ii},:)  ;
    tr_y=vertcat(y_tr([ids{[1:ii-1 ii+1:k]}],:));
    
    K=K_tr;% Create temporary K for k-1 training folds
    K(ids{ii},:)=[];
    K(:,ids{ii})=[];
    
    alpha = kridgereg( K,tr_y,gamma );
    K=K_tr(ids{ii},:);% Create temporary K for the testing
    K(:,ids{ii})=[];
    scores  = dualcost( K,ev_y,alpha );
end
score=mean(scores);
end

%This function randomly splits a given sample set and ground truth y, to 2:1 ratio for training and testing sets.
function [X_train, y_train, X_test,y_test]=splitData(X,y)
%% Shuffle data
ids=randperm(size(X,1));
X=X(ids,:);
y=y(ids);
%% 2:1 ratio split hardcoded
split=2/3;
%% divide data
X_train=X(1:floor(size(X,1)*split),:);
y_train=y(1:floor(size(X,1)*split));
X_test=X(floor(size(X,1)*split)+1:end,:);
y_test=y(floor(size(X,1)*split)+1:end);
end

%This function computes a Kernel matrix such that the K_{i,j} entry is the kernel function k(X_i,T_j), where X,T some sets of samples.
% Thus Kernel_mat2(train,train,v) would outout K_{train,train}. sigma is
% the parameter for  the gaussian kernel function
function K = Kernel_mat2(X,T,sigma)
[m,n]=size(X);
[mt,nt]=size(T);
K=zeros(m,mt);
for ii=1:m
    % Vectorised
    K(ii,:)=  exp(-sum((repmat(X(ii,:),mt,1)-T).^2,2) ./(2*sigma^2)) ;
end
end