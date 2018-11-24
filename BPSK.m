function BPSK()

N = 1000000;
SNR = 20;
rng('shuffle');
% Generate the Original Data
Sorg = randi(2,1,N)*2-3;
% Add Noise
Sn = awgn(Sorg, SNR+3);
Sn = Sn/sqrt(Sn*Sn'/N);

inputs = Sn;
targets = Sorg;
 
% Create a Fitting Network
hiddenLayerSize = 2;
net = fitnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
 
% Train the Network
[net,tr] = train(net,inputs,targets);

SNR = 5:20;
mse = zeros(1,length(SNR));
for SNRidx = 1:length(SNR);
    % Generate the Original Data
    Sorg = randi(2,1,N)*2-3;
    % Add Noise
    Sn = awgn(Sorg, SNR(SNRidx)+3);
    Sn = Sn/sqrt(Sn*Sn'/N);

    inputs = Sn;
    targets = Sorg;

    % Test the Network
    outputs = net(inputs);
    errors = gsubtract(outputs,targets);
    performance = perform(net,targets,outputs);
    out = sign(outputs);
    mse(SNRidx) = (N-length(find(out - targets == 0)))/N;
    mse(SNRidx)
end;

semilogy(SNR, mse, '-*'); grid on;
 
end

