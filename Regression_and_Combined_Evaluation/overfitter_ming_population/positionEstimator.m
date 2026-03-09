function [decodedPosX, decodedPosY, modelParameters] = positionEstimator(testData, modelParameters)
% Direction from POPULATION VECTOR (decode angle from sum(rate * prefDir)); then velocity regression + blend.

bufferBinsLong  = modelParameters.bufferSizeLong;
bufferBinsShort = modelParameters.bufferSizeShort;
dirWindowEnd    = modelParameters.dirWindowEnd;
smoothKernel    = modelParameters.smoothKernel;
prefVec         = modelParameters.prefVec;   % 98 x 2

raw = testData.spikes;
cleaned = conv2(1, smoothKernel, raw, 'same');

trialId = double(testData.trialId);

% --------- Direction: population vector (once per trial, cached) ---------
if isKey(modelParameters.trialDirMap, trialId)
    d = modelParameters.trialDirMap(trialId);
else
    Tend = min(size(cleaned,2), dirWindowEnd);
    featEarly = sum(cleaned(:,1:Tend), 2)';   % 1 x 98
    % Population vector = weighted sum of preferred directions
    popVec = featEarly * prefVec;             % 1 x 2
    angle = atan2(popVec(2), popVec(1));      % [-pi, pi]
    % Map to discrete direction 1..8 (0°, 45°, ..., 315°)
    d = mod(round(angle / (pi/4)), 8) + 1;
    if d < 1, d = 1; end
    if d > 8, d = 8; end
    modelParameters.trialDirMap(trialId) = d;
end

% --------- Multi-scale buffer, z-score, velocity decode, integrate ---------
T = size(cleaned, 2);
startLong  = T - bufferBinsLong  + 1;
startShort = T - bufferBinsShort + 1;
if startLong  < 1, startLong  = 1; end
if startShort < 1, startShort = 1; end

featLong  = sum(cleaned(:, startLong:T), 2)';
featShort = sum(cleaned(:, startShort:T), 2)';
feat = [featShort, featLong];

mu = modelParameters.featMean{d};
sig = modelParameters.featStd{d};
sig(sig < 1e-6) = 1;
featN = (feat - mu) ./ sig;
featB = [featN 1];

decodedVelX = featB * modelParameters.linearModelVx{d};
decodedVelY = featB * modelParameters.linearModelVy{d};

if isempty(testData.decodedHandPos)
    lastX = testData.startHandPos(1);
    lastY = testData.startHandPos(2);
else
    lastX = testData.decodedHandPos(1, end);
    lastY = testData.decodedHandPos(2, end);
end

decodedPosX = lastX + decodedVelX;
decodedPosY = lastY + decodedVelY;

% Blend with direction-specific average trajectory
avgTraj = modelParameters.avgTraj{d};
wAvg = modelParameters.avgTrajWeight;
tIdx = min(T, size(avgTraj, 2));
decodedPosX = (1 - wAvg) * decodedPosX + wAvg * avgTraj(1, tIdx);
decodedPosY = (1 - wAvg) * decodedPosY + wAvg * avgTraj(2, tIdx);
end
