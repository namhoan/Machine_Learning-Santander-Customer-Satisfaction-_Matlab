clc
clear all
close all

% =========================================================================
%                   Classification using Decision Tree
% =========================================================================

% Training.

% Read Training data.

fprintf('Reading Training Data . . .\n')
trdata = csvread('train.csv',1,0);
fprintf('\nTraining data loaded.\n')

% Separate training data.

datanum = trdata(:,1);
inputs = trdata(:,2:end-1);
target = trdata(:,end);

I = find(target == 0);

I1 = randi(length(I),100);
I1 = I(I1);

I = find(target == 1);
I2 = randi(length(I),100);
I2 = I(I2);
I = sort([I1,I2]);

inputs = inputs(I,:)';
target = target(I,:)';

clear('trdata')

net = newpr(inputs,target,100); 
net.layers{1}.transferFcn='tansig';  

net.layers{2}.transferFcn='purelin';  
net.trainParam.epochs = 100;  
net = train(net,inputs,double(target)); 

fprintf('\nReading test data . . . \n')
tedata = csvread('test.csv',1,0);
fprintf('\nTest data loaded. \n')

testdatanum = tedata(:,1);
testinputs = tedata(:,2:end);

res = sim(net,testinputs');
res = res';
res = res >= 0;
res = double(res);
