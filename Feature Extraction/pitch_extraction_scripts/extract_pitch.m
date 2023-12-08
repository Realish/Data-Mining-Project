genres = ["blues" "classical" "country" "disco" "hiphop" "jazz" "metal" "pop" "reggae" "rock"];

for foldername = genres
    %foldername = 'blues';
    pathstr = strcat('../genres_original/', foldername, '/*.wav');
    pathtofile = strcat('../genres_original/', foldername);
    
    fileList = dir(pathstr);
    fileList = {fileList.name};
    
    disp(pathtofile)
    
    
    for wavfilename = fileList
        wavfilename = string(wavfilename);
        parameter=[];
        parameter.useResampling = 1;
        parameter.destSamplerate = 22050;
        parameter.convertToMono = 1;
        parameter.monoConvertMode = 'downmix';
        parameter.message = 0;
        parameter.vis = 0;
        parameter.save = 1;
        dirAbs = '';
        dirRel = 'audio/';
        parameter.saveDir = [dirAbs,dirRel];
        
        %wavfilename = 'classical.00000.wav';
        [pathstr,name,ext] = fileparts(wavfilename);
        [f_audio, sideinfo] = wav_to_audio(pathtofile,'/',wavfilename,parameter);
        
        
        parameter=[];
        parameter.winLenSTMSP = 4410;
        parameter.shiftFB = 0;
        parameter.midiMin = 21;
        parameter.midiMax = 108;
        parameter.save = 1;
        parameter.saveDir = strcat('pitch_folder/', foldername, '/');
        parameter.saveFilename = name;
        parameter.saveAsTuned = 0;
        parameter.fs = 22050;
        parameter.visualize = 1;
        
        [f_pitch,sideinfo] = audio_to_pitch_via_FB(f_audio,parameter,sideinfo);
        disp(f_pitch)
    
    end
end
