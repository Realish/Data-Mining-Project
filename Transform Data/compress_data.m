% testThis is code to add compression to each audio file in the GTZAN dataset.

% frame length is related to the sample rate
frameLength = 1024;

% loop through and add compression to each file
audioFolder = 'Data';
wavFiles = dir(fullfile(audioFolder, '*.wav'));
for i = 1:length(wavFiles)
    wavFile = fullfile(audioFolder,wavFiles(i).name);
    fileName = wavFiles(i).name;
    outputFilePath = fullfile('Compressed_Data', ['compressed_' fileName]);

    % read file
    fileReader = dsp.AudioFileReader( ...
        'Filename',wavFile, ...
        'SamplesPerFrame',frameLength);

    % % initialize the object that plays audio data using computers audio device
    % deviceWriter = audioDeviceWriter( ...
    %     'SampleRate',fileReader.SampleRate);

    % initialize dynamic range compressor
    % parameters: threshold, ratio, knee width, sample rate
    dRC = compressor(-20,10, ...
        'KneeWidth',5, ...
        'SampleRate',fileReader.SampleRate);

    % % visualize the audio
    % scope = timescope( ...
    %     'SampleRate',fileReader.SampleRate, ...
    %     'TimeSpanSource','Property','TimeSpan',1, ...
    %     'BufferLength',44100*4, ...
    %     'YLimits',[-1,1], ...
    %     'TimeSpanOverrunAction','Scroll', ...
    %     'ShowGrid',true, ...
    %     'LayoutDimensions',[2,1], ...
    %     'NumInputPorts',2, ...
    %     'Title', ...
    %     ['Original vs. Compressed Audio (top)' ...
    %     ' and Compressor Gain in dB (bottom)']);
    % scope.ActiveDisplay = 2;
    % scope.YLimits = [-10,0];
    % scope.YLabel = 'Gain (dB)';

    % apply compressor to the audio
    % y is compressed audio data, g is gain
    compressedAudioData = [];

    while ~isDone(fileReader)
        x = fileReader();
        [y,g] = dRC(x);
        %deviceWriter(y);
        %scope([x(:,1),y(:,1)],g(:,1))
        compressedAudioData = [compressedAudioData; y];
    end

    release(dRC);
    release(fileReader);
    %release(deviceWriter)
    %release(scope)

    audiowrite(outputFilePath, compressedAudioData, fileReader.SampleRate);
end