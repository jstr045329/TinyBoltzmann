% This script uses 2 dummy 
SHOW_TRAINING_DATA = 0;

% Make a short fat rectangle:
img_01 = zeros(15);
for i = [6, 8]
    img_01(i, 3:13) = ones(1, 11);
end
img_01(7, 3) = 1;
img_01(7, 13) = 1;

% Make a tall skinny rectangle:
img_02 = img_01';

if SHOW_TRAINING_DATA
    heatmap(img_02);
end

layerVec = [225, 160];
uut = TinyBoltzmann(layerVec, 0.1, 0.99, 0.05);

trainingSet = {};
trainingSet{1} = img_01;
trainingSet{2} = img_02;

%uut = uut.pretrainLayer(1, trainingSet, 10);

% function obj = pretrainNTimes(obj, lowerNum, trainingSet, innerIterations, outerIterations, enableTrim)
uut = uut.pretrainNTimes(1, trainingSet, 10, 100, 1);
uut = uut.pretrainNTimes(1, trainingSet, 5, 100, 1);
%uut = uut.pretrainNTimes(1, trainingSet, 5, 500, 1);

