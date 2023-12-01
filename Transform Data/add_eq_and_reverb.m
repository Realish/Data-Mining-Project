%setup(deviceWriter,ones(frameLength,2))

% This is code to add equalisation to each audio file in the GTZAN dataset.

% frame length is related to the sample rate
frameLength = 1024;

% loop through and add compression to each file
audioFolder = 'Data';
wavFiles = dir(fullfile(audioFolder, '*.wav'));
for i = 1:length(wavFiles)
    wavFile = fullfile(audioFolder,wavFiles(i).name);
    fileName = wavFiles(i).name;
    outputFilePath = fullfile('Highpass_Data', ['highpass_' fileName]);

    % read file
    fileReader = dsp.AudioFileReader( ...
        'Filename',wavFile, ...
        'SamplesPerFrame',frameLength);

    % initialize equalisation, lowpass and highpass
    % mPEQ = multibandParametricEQ( ...
    % 'NumEQBands',3, ...
    % 'Frequencies',[300,1200,5000], ...
    % 'QualityFactors',[1,1,1], ...
    % 'PeakGains',[1,-20,-20], ...
    % 'HasLowpassFilter',true, ... % Enable low-pass filter
    % 'HasHighpassFilter',false, ...  % Disable high-pass filter
    % 'LowpassCutoff', 500, ...   % Set low-pass filter cutoff frequency
    % 'SampleRate',fileReader.SampleRate);

    mPEQ = multibandParametricEQ( ...
    'NumEQBands',3, ...
    'Frequencies',[300,1200,5000], ...
    'QualityFactors',[1,1,1], ...
    'PeakGains',[0,0,0], ...
    'HasHighpassFilter',true, ...  % Enable high-pass filter
    'HasLowpassFilter',false, ...  % Disable low-pass filter
    'HighpassCutoff', 500, ...     % Set high-pass filter cutoff frequency
    'SampleRate',fileReader.SampleRate);

    equalisedAudioData = [];

    while ~isDone(fileReader)
        x = fileReader();
        y = mPEQ(x);
        equalisedAudioData = [equalisedAudioData; y];
    end

    release(mPEQ);
    release(fileReader);

    audiowrite(outputFilePath, equalisedAudioData, fileReader.SampleRate);
end