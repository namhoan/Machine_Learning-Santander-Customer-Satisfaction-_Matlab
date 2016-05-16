clc
clear all
close all

% Neural network based solution.

load net21;

fprintf('\nReading Training data . . . \n')
tedata = csvread('train.csv',1,0);
fprintf('\nTest data loaded. \n')

% Separate training data.

datanum = trdata(:,1);
inputs = trdata(:,2:end-1);
target = trdata(:,end);

% Performance on self
output = sim(net, inputs');

prune = output';
i = size(prune,1);
while i > 0
    if(target(i) == 0)
        prune(i) = [];
    end
    i = i-1;
end

E = floor(100*median(prune))/100;

output2 = double(output' >= E);
%[c,cm] = confusion(target,output);
off = sum(target == output2)/size(target, 1);
fprintf('\n%d\n', off);

figure;
n_bins = 1000;
start = floor(min(output')*n_bins)/n_bins;
ending = floor(max(output')*n_bins)/n_bins;
bins = start:(ending-start)/n_bins:ending;
hist(output',bins);
h1 = findobj(gca,'Type','patch');
set(h1,'FaceColor','r','EdgeColor','k'); % Alt blue: [.4,.7,1]
hold on;
hist(prune,bins);
h2 = findobj(gca,'Type','patch');
%set(h2,'FaceColor',[.4,.7,1],'EdgeColor','k');
hold off;

% Performance on test

fprintf('\nReading test data . . . \n')
tedata = csvread('test.csv',1,0);
fprintf('\nTest data loaded. \n')

testdatanum = tedata(:,1);
testinputs = tedata(:,2:end);

res = sim(net,testinputs');
res = double(res' >= E);

result = uint32([testdatanum,res]);

csvwrite('ann_result.csv',result)