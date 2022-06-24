#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or later
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Designed to suppress noise in wav audio using the RNNoise library (https://github.com/xiph/rnnoise).

Contains the RNNoise class. Read more at https://github.com/Desklop/RNNoise_Wrapper.

Dependencies: pydub, numpy.
'''

import os
import platform
import time
import ctypes
import pkg_resources
import numpy as np
from pydub import AudioSegment


__version__ = 1.1


class RNNoise(object):
    """Provides methods to simplify working with RNNoise:
    - read_wav(): loading a .wav audio recording and converting it to a supported format
    - write_wav(): save .wav audio recording
    - filter(): split audio into frames and clean them from noise
    - filter_frame(): clearing only one frame from noise (directly accessing the RNNoise binary)
    - reset(): recreate the RNNoise object from the library to reset the state of the neural network

    1. f_name_lib - path to the library, if None and:
            - OS type linux or mac (darwin) - use librnnoise_5h_b_500k.so.0.4.1 from package files
            - the type of OS used windows or other - search in the current folder and its subfolders for a file with the prefix 'librnnoise'
        if is a path to a library/library name - check the existence of the passed path/library name and if:
            - path/name exists - return absolute path
            - path/name does not exist - search the current folder and its subfolders for the file/path using the passed value as the subname
    """
    sample_width = 2
    channels = 1
    sample_rate = 48000
    frame_duration_ms = 10

    def __init__(self, f_name_lib=None):
        f_name_lib = self.__get_f_name_lib(f_name_lib)
        self.rnnoise_lib = ctypes.cdll.LoadLibrary(f_name_lib)

        self.rnnoise_lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self.rnnoise_lib.rnnoise_process_frame.restype = ctypes.c_float
        self.rnnoise_lib.rnnoise_create.restype = ctypes.c_void_p
        self.rnnoise_lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]

        self.rnnoise_obj = self.rnnoise_lib.rnnoise_create(None)


    def __get_f_name_lib(self, f_name_lib=None):
        '''Find and/or check the path to the compiled RNNoise library.

        1. f_name_lib - path to the library, if None and:
                - OS type linux or mac (darwin) - use librnnoise_5h_b_500k.so.0.4.1 from package files
                - the type of OS used windows or other - search in the current folder and its subfolders for a file with the prefix 'librnnoise'
            if is a path to a library/library name - check the existence of the passed path/library name and if:
                - path/name exists - return absolute path
                - path/name does not exist - search the current folder and its subfolders for the file/path using the passed value as the subname
        2. returns f_name_lib with checked absolute path to found library'''

        package_name = __file__
        package_name = package_name[package_name.rfind('/')+1:package_name.rfind('.py')]

        if not f_name_lib:
            subname = 'librnnoise'
            system = platform.system()
            if system == 'Linux' or system == 'Darwin':
                found_f_name_lib = pkg_resources.resource_filename(package_name, 'libs/{}_5h_b_500k.so.0.4.1'.format(subname))
                if not os.path.exists(found_f_name_lib):
                    found_f_name_lib = self.__find_lib(subname)
            else:
                found_f_name_lib = self.__find_lib(subname)

            if not found_f_name_lib:
                raise NameError("could not find RNNoise library with subname '{}'".format(subname))

        else:
            f_names_available_libs = pkg_resources.resource_listdir(package_name, 'libs/')
            for available_lib in f_names_available_libs:
                if available_lib.find(f_name_lib) != -1:
                    f_name_lib = pkg_resources.resource_filename(package_name, 'libs/{}'.format(available_lib))

            found_f_name_lib = self.__find_lib(f_name_lib)
            if not found_f_name_lib:
                raise NameError("could not find RNNoise library with name/subname '{}'".format(f_name_lib))

        return found_f_name_lib


    def __find_lib(self, f_name_lib, root_folder='.'):
        ''' Perform a recursive search for the f_name_lib file in the root_folder and all its subfolders.
        1. f_name_lib - the name of the file being searched for or its subname (part of the name that allows you to uniquely identify the file)
        2. root_folder - root folder from which to start searching
        3. returns found existing path or None'''

        f_name_lib_full = os.path.abspath(f_name_lib)
        if os.path.isfile(f_name_lib_full) and os.path.exists(f_name_lib_full):
            return f_name_lib_full

        for path, folder_names, f_names in os.walk(root_folder):
            for f_name in f_names:
                if f_name.rfind(f_name_lib) != -1:
                    return os.path.join(path, f_name)


    def reset(self):
        '''Reset the state of the neural network by creating a new RNNoise object in the compiled source library.
        Can be useful when noise reduction is used on a large number of audio recordings to prevent degradation
        work quality.

        The effectiveness and necessity of this method has not been proven. Implemented just in case :)'''

        self.rnnoise_lib.rnnoise_destroy(self.rnnoise_obj)
        self.rnnoise_obj = self.rnnoise_lib.rnnoise_create(None)


    def filter_frame(self, frame):
        '''Denoising one frame with RNNoise. The frame must be 10 milliseconds long in 16 bit 48 kHz format.
        1. frame - byte string with audio data
        2. returns a tuple from the probability of having a voice in the frame and a denoised frame

        The probability of having a vote in a frame (called 'vad_probability' in denoise.c) is a number between 0 and 1 representing the probability
        that the frame contains a voice (or perhaps a loud sound). Can be used to implement an inline VAD.'''

        # 480 = len(frame)/2, len(frame) should always be 960 values ​​(because frame width is 2 bytes (16 bits))
        # (i.e. frame length 10 ms (0.01 sec) at 48000 Hz sample rate, 48000*0.01*2=960).
        # If len(frame) != 960, there will be a segmentation error or severe distortion in the final audio recording.

        # If we move np.ndarray((480,), 'h', frame).astype(ctypes.c_float) to __get_frames(), then the performance gain will be
        # no more than 5-7% on audio recordings, longer than 60 seconds. On shorter audio recordings, the increase in speed is less noticeable and insignificant

        frame_buf = np.ndarray((480,), 'h', frame).astype(ctypes.c_float)
        frame_buf_ptr = frame_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        vad_probability = self.rnnoise_lib.rnnoise_process_frame(self.rnnoise_obj, frame_buf_ptr, frame_buf_ptr)
        return vad_probability, frame_buf.astype(ctypes.c_short).tobytes()


    def filter(self, audio, sample_rate=None, voice_prob_threshold=0.0, save_source_sample_rate=True):
        ''' Get frames from an audio recording and de-noise them. RNNoise is used for noise reduction.

        RNNoise additionally for each frame returns the probability of having a vote in this frame (as a number from 0 to 1) and
        using voice_prob_threshold, you can filter frames by this value. If the probability is lower than voice_prob_threshold,
        then the frame will be removed from the audio recording.

        ATTENTION! The sampling rate of the audio recording is forced to 48 kHz. Other values ​​are not supported by RNNoise.
        However, the sample rate of the returned audio can be downsampled to the original.

        ATTENTION! For RNNoise to work successfully, you need an audio recording of at least 1 second in length, which contains both voice and noise.
        (Moreover, the noise should ideally be the same in front of the voice). Otherwise, the quality of noise reduction will be very poor.

        ATTENTION! If parts of one audio recording are transmitted (audio noise reduction in the stream), then their length must be at least 10 ms
        and a multiple of 10 (because the RNNoise library only supports 10ms frames). This option works on the quality of noise reduction
        practically no effect.

        1. audio - pydub.AudioSegment object with audio recording or byte string with audio data (no wav headers)
        2. sample_rate - sample rate (required only when audio is a byte string)
        3. voice_prob_threshold - threshold for the probability of having a voice in each frame (value from 0 to 1, if 0 - use all frames)
        4. save_source_sample_rate - True: bring the sample rate of the returned audio recording to the original
        5. returns pydub.AudioSegment or a byte string (without wav headers) denoised (the returned object type is audio)'''

        frames, source_sample_rate = self.__get_frames(audio, sample_rate)
        if not save_source_sample_rate:
            source_sample_rate = None

        denoised_audio = self.__filter_frames(frames, voice_prob_threshold, source_sample_rate)

        if isinstance(audio, AudioSegment):
            return denoised_audio
        else:
            return denoised_audio.raw_data


    def __filter_frames(self, frames, voice_prob_threshold=0.0, sample_rate=None):
        ''' Clearing frames from noise. RNNoise is used for noise reduction.

        RNNoise additionally for each frame returns the probability of having a vote in this frame (as a number from 0 to 1) and
        using voice_prob_threshold, you can filter frames by this value. If the probability is lower than voice_prob_threshold,
        then the frame will be removed from the audio recording.

        ATTENTION! For RNNoise to work successfully, you need an audio recording of at least 1 second in length, which contains both voice and noise.
        (Moreover, the noise should ideally be the same in front of the voice). Otherwise, the quality of noise reduction will be very poor.

        ATTENTION! If parts of one audio recording are transmitted, then their length must be at least 10 ms (the length of one frame) and a multiple of 10
        (because RNNoise only supports 10ms frames). This option does not affect the quality of noise reduction and can be used
        to denoise the audio in the stream.

        1. frames - a list of frames with a length of 10 milliseconds
        2. voice_prob_threshold - threshold for the probability of having a voice in each frame (value from 0 to 1, if 0 - use all frames)
        3. sample_rate - the desired sampling rate of the cleared audio recording (if None - do not change the sampling rate)
        4. returns a pydub.AudioSegment object with the denoised audio recording'''

        denoised_frames_with_probability = [self.filter_frame(frame) for frame in frames]
        denoised_frames = [frame_with_prob[1] for frame_with_prob in denoised_frames_with_probability if frame_with_prob[0] >= voice_prob_threshold]
        denoised_audio_bytes = b''.join(denoised_frames)

        denoised_audio = AudioSegment(data=denoised_audio_bytes, sample_width=self.sample_width, frame_rate=self.sample_rate, channels=self.channels)

        if sample_rate:
            denoised_audio = denoised_audio.set_frame_rate(sample_rate)
        return denoised_audio


    def __get_frames(self, audio, sample_rate=None):
        '''Get frames from an audio recording. Frames are byte strings of fixed length audio data.
        RNNoise only supports 10 millisecond frames.

        ATTENTION! The sampling rate of the audio recording is forced to 48 kHz. Other values ​​are not supported by RNNoise.

        1. audio - pydub.AudioSegment object with audio recording or byte string with audio data (no wav headers)
        2. sample_rate - sample rate (required only when audio is a byte string):
            if the sampling rate is not supported - it will be converted to supported 48 kHz
        3. returns a tuple from a list of frames and the original sample rate of the audio recording
        '''

        if isinstance(audio, AudioSegment):
            sample_rate = source_sample_rate = audio.frame_rate
            if sample_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)
            audio_bytes = audio.raw_data
        elif isinstance(audio, bytes):
            if not sample_rate:
                raise ValueError("when type(audio) = 'bytes', 'sample_rate' can not be None")
            audio_bytes = audio
            source_sample_rate = sample_rate
            if sample_rate != self.sample_rate:
                audio = AudioSegment(data=audio_bytes, sample_width=self.sample_width, frame_rate=sample_rate, channels=self.channels)
                audio = audio.set_frame_rate(self.sample_rate)
                audio_bytes = audio.raw_data
        else:
            raise TypeError("'audio' can only be AudioSegment or bytes")

        frame_width = int(self.sample_rate * (self.frame_duration_ms / 1000.0) * 2)
        if len(audio_bytes) % frame_width != 0:
            silence_duration = frame_width - len(audio_bytes) % frame_width
            audio_bytes += b'\x00' * silence_duration

        offset = 0
        frames = []
        while offset + frame_width <= len(audio_bytes):
            frames.append(audio_bytes[offset:offset + frame_width])
            offset += frame_width
        return frames, source_sample_rate


    def read_wav(self, f_name_wav, sample_rate=None):
        '''Download .wav audio recording. Only 2-byte/16-bit mono audio recordings are supported. If the parameters of the downloaded audio recording
        different from those specified - it will be converted to the required format.
        1. f_name_wav - name of the .wav audio recording or BytesIO
        2. sample_rate - the desired sampling rate (if None - do not change the sampling rate)
        3. returns a pydub.AudioSegment object with an audio recording'''

        if isinstance(f_name_wav, str) and f_name_wav.rfind('.wav') == -1:
            raise ValueError("'f_name_wav' must contain the name .wav audio recording")

        audio = AudioSegment.from_wav(f_name_wav)

        if sample_rate:
            audio = audio.set_frame_rate(sample_rate)
        if audio.sample_width != self.sample_width:
            audio = audio.set_sample_width(self.sample_width)
        if audio.channels != self.channels:
            audio = audio.set_channels(self.channels)
        return audio


    def write_wav(self, f_name_wav, audio_data, sample_rate=None):
        '''Save .wav audio recording.
        1. f_name_wav - the name of the .wav audio recording where the audio recording or BytesIO will be saved
        2. audio_data - pydub.AudioSegment object with audio recording or byte string with audio data (no wav header)
        3. sample_rate - audio sample rate:
            when audio_data is a byte string, must match the actual sample rate of the audio recording
            in other cases, the sampling rate will be reduced to the specified one (if None - do not change the sampling rate)'''

        if isinstance(audio_data, AudioSegment):
            self.write_wav_from_audiosegment(f_name_wav, audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            if not sample_rate:
                raise ValueError("when type(audio_data) = 'bytes', 'sample_rate' can not be None")
            self.write_wav_from_bytes(f_name_wav, audio_data, sample_rate)
        else:
            raise TypeError("'audio_data' is of an unsupported type. Supported:\n" + \
                            "\t- pydub.AudioSegment with audio\n" + \
                            "\t- byte string with audio data (without wav header)")


    def write_wav_from_audiosegment(self, f_name_wav, audio, desired_sample_rate=None):
        '''Save .wav audio recording.
        1. f_name_wav - the name of the .wav file where the audio recording or BytesIO will be saved
        2. audio - pydub.AudioSegment object with audio recording
        3. desired_sample_rate - the desired sample rate (if None - do not change the sample rate)'''

        if desired_sample_rate:
            audio = audio.set_frame_rate(desired_sample_rate)
        audio.export(f_name_wav, format='wav')


    def write_wav_from_bytes(self, f_name_wav, audio_bytes, sample_rate, desired_sample_rate=None):
        '''Save .wav audio recording.
        1. f_name_wav - the name of the .wav file where the audio recording or BytesIO will be saved
        2. audio_bytes - byte string with audio recording (without wav headers)
        3. sample_rate - sample rate
        4. desired_sample_rate - the desired sample rate (if None - do not change the sample rate)'''

        audio = AudioSegment(data=audio_bytes, sample_width=self.sample_width, frame_rate=sample_rate, channels=self.channels)
        if desired_sample_rate and desired_sample_rate != sample_rate:
            audio = audio.set_frame_rate(desired_sample_rate)

        audio.export(f_name_wav, format='wav')




def main():
    folder_name_with_audio = 'test_audio/functional_tests'
    f_name_rnnoise_binary = 'librnnoise_default.so.0.4.1'

    denoiser = RNNoise(f_name_rnnoise_binary)

    # Search audio recordings for test
    all_objects = os.listdir(folder_name_with_audio)
    f_names_source_audio = []
    for one_object in all_objects:
        if os.path.isfile(os.path.join(folder_name_with_audio, one_object)) and one_object.rfind('.wav') != -1 \
                                                                            and one_object.rfind('denoised') == -1:
            f_names_source_audio.append(os.path.join(folder_name_with_audio, one_object))
    f_names_source_audio = sorted(f_names_source_audio, key=lambda f_name: int(f_name[f_name.rfind('_')+1:f_name.rfind('.')]))


    # Test for working with audio as a byte string without headers
    f_name_audio = f_names_source_audio[0]
    audio = denoiser.read_wav(f_name_audio)

    start_time = time.time()
    denoised_audio = denoiser.filter(audio.raw_data, sample_rate=audio.frame_rate)
    elapsed_time = time.time() - start_time

    f_name_denoised_audio = f_name_audio[:f_name_audio.rfind('.wav')] + '_denoised.wav'
    denoiser.write_wav(f_name_denoised_audio, denoised_audio, sample_rate=audio.frame_rate)

    print("Audio: '{}', length: {:.2f} s:".format(f_name_audio, len(audio)/1000))
    print("\tdenoised audio    '{}'".format(f_name_denoised_audio))
    print('\tprocessing time   {:.2f} s'.format(elapsed_time))
    print('\tprocessing speed  {:.1f} RT'.format(len(audio)/1000/elapsed_time))


    denoiser.reset()  # not necessarily, need has not yet been proven


    # Test for working with streaming audio (buffer size 10 ms = 1 frame) - processing audio recording for 10 milliseconds
    f_name_audio = f_names_source_audio[0]
    audio = denoiser.read_wav(f_name_audio)

    denoised_audio = b''
    buffer_size_ms = 10

    start_time = time.time()
    elapsed_time_per_frame = []
    for i in range(buffer_size_ms, len(audio), buffer_size_ms):
        time_per_frame = time.time()
        denoised_audio += denoiser.filter(audio[i-buffer_size_ms:i].raw_data, sample_rate=audio.frame_rate)
        elapsed_time_per_frame.append(time.time() - time_per_frame)
    if len(audio) % buffer_size_ms != 0:
        time_per_frame = time.time()
        denoised_audio += denoiser.filter(audio[len(audio)-(len(audio)%buffer_size_ms):].raw_data, sample_rate=audio.frame_rate)
        elapsed_time_per_frame.append(time.time() - time_per_frame)
    elapsed_time = time.time() - start_time
    average_elapsed_time_per_frame = sum(elapsed_time_per_frame) / len(elapsed_time_per_frame)

    f_name_denoised_audio = f_name_audio[:f_name_audio.rfind('.wav')] + '_denoised_stream.wav'
    denoiser.write_wav(f_name_denoised_audio, denoised_audio, sample_rate=audio.frame_rate)

    print("\nAudio: '{}', length: {:.2f} s:".format(f_name_audio, len(audio)/1000))
    print("\tdenoised audio                                '{}'".format(f_name_denoised_audio))
    print('\tprocessing time                               {:.2f} s'.format(elapsed_time))
    print('\taverage processing time of 1 buffer ({} ms)   {:.2f} ms'.format(buffer_size_ms, average_elapsed_time_per_frame*1000))
    print('\tprocessing speed                              {:.1f} RT'.format(len(audio)/1000/elapsed_time))
    print('\taverage processing speed of 1 buffer ({} ms)  {:.1f} RT'.format(buffer_size_ms, buffer_size_ms/(average_elapsed_time_per_frame*1000)))


    # Test for working with audio in the form of pydub.AudioSegment
    for f_name_audio in f_names_source_audio[1:6]:
        audio = denoiser.read_wav(f_name_audio)

        start_time = time.time()
        denoised_audio = denoiser.filter(audio)
        elapsed_time = time.time() - start_time

        f_name_denoised_audio = f_name_audio[:f_name_audio.rfind('.wav')] + '_denoised.wav'
        denoiser.write_wav(f_name_denoised_audio, denoised_audio)

        print("\nAudio: '{}', length: {:.2f} s:".format(f_name_audio, len(audio)/1000))
        print("\tdenoised audio    '{}'".format(f_name_denoised_audio))
        print('\tprocessing time   {:.2f} s'.format(elapsed_time))
        print('\tprocessing speed  {:.1f} RT'.format(len(audio)/1000/elapsed_time))


if __name__ == '__main__':
    main()
