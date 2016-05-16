clc
clear all
close all

% Load dataset

fprintf('Reading Training Data . . .\n')
trdata = csvread('train.csv',1,0);
fprintf('\nTraining data loaded.\n')

% Separate training data.

datanum = trdata(:,1);
inputs = trdata(:,2:end-1);
target = trdata(:,end);

% Train network
% TODO?: Determine ideal hidden neurons (3rd value of newpr)
% Currently ~1/100th of the input size (76021)
% TODO: Minimize reduction according to memory capacity (N).
% 'useParallel' possibly?
net = patternnet(700);
%net.layers{1}.transferFcn='tansig';
%net.layers{2}.transferFcn='purelin';
N = 100;
net.trainParam.epochs = 1000; % (default 1000?)
net.trainParam.showWindow = false;          % no GUI, for server use
net.trainParam.showCommandLine = true;      % display in command line
net.trainParam.show = 1;                    % display every iteration
net = train(net, inputs', double(target)', 'reduction', N);

% Save trained network for future possible test use

save net;

% Performance on self
output = sim(net, inputs');
output = double(output' > 0);
%[c,cm] = confusion(target,output);
off = sum(target == output)/size(target, 1);
fprintf('\n%d\n', off);

% Performance on test

fprintf('\nReading test data . . . \n')
tedata = csvread('test.csv',1,0);
fprintf('\nTest data loaded. \n')

testdatanum = tedata(:,1);
testinputs = tedata(:,2:end);

res = sim(net,testinputs');
res = double(res' > 0);

result = uint32([testdatanum,res]);

csvwrite('ann_result.csv',result)