# Fork Description

Personal fork of [RNNoise_Wrapper](https://github.com/Desklop/RNNoise_Wrapper). This fork was actually forked from [another fork](https://github.com/sidneydavis/RNNoise_Wrapper), because it contains docs and comments translated to English (originally in Russian). [This doc](https://www.big-meter.com/opensource/en/5fe76682e90e736ddf283013.html) has more info about RNNoise. Below is the original translated README.

------------

# RNNoise Wrapper

This is a simple Python wrapper for noise reduction [RNNoise](https://github.com/xiph/rnnoise). Only Python 3 is supported. The code is based on [issue by snakers4](https://github.com/xiph/rnnoise/issues/69) from the RNNoise repository, for which special thanks to him.

[RNNoise](https://jmvalin.ca/demo/rnnoise/) is a recurrent neural network with GRU cells for real-time audio noise reduction (even works on Raspberry Pi). The standard model is trained on 6.4GB of noisy audio recordings and is ready to use.

RNNoise is written in C and has methods for denoising a single frame of 10 milliseconds. The frame must be 48000Hz, mono, 16 bits.

**RNNoise_Wrapper** makes it easy to work with RNNoise:

- eliminates the need to extract frames / frames from the audio recording yourself
- removes restrictions on the parameters of the processed wav audio recording
- hides all the nuances of working with the C library
- eliminates the need to manually compile RNNoise (Linux only)
- adds 2 new binaries with better models that come with the package (Linux only)

**RNNoise_Wrapper contains 2 new better models** (trained weights and compiled RNNoise binaries for Linux). The dataset from [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge) was used for training.

1. **librnnoise_5h_ru_500k** - trained on 5 hours of Russian speech (with emotional speech and singing mixed in English), obtained by a script from a dataset repository. The trained weights are in [`train_logs/weights_5h_ru_500k.hdf5`](https://github.com/Desklop/RNNoise_Wrapper/tree/master/train_logs/weights_5h_ru_500k.hdf5) compiled by RNNoise in [`rnnoise_wrapper/libs/librnnoise_5h_ru_500k.so. 0.4.1`](https://github.com/Desklop/RNNoise_Wrapper/tree/master/rnnoise_wrapper/libs/librnnoise_5h_ru_500k.so.0.4.1) (Linux only)

2. **librnnoise_5h_b_500k** - trained in 5 hours of mixed speech in English, Russian, German, French, Italian, Spanish and Mandarin Chinese (with emotional speech and singing mixed in English). The dataset for each language was previously trimmed by the smallest of them (the smallest data for the Russian language, about 47 hours). The final training sample was obtained by the script from the repository with the dataset. The trained weights are in [`train_logs/weights_5h_b_500k.hdf5`](https://github.com/Desklop/RNNoise_Wrapper/tree/master/train_logs/weights_5h_b_500k.hdf5) compiled by RNNoise in [`rnnoise_wrapper/libs/librnnoise_5h_b_500k.so. 0.4.1`](https://github.com/Desklop/RNNoise_Wrapper/tree/master/rnnoise_wrapper/libs/librnnoise_5h_b_500k.so.0.4.1) (Linux only)

3. **librnnoise_default** - standard model from the authors of [RNNoise] (https://jmvalin.ca/demo/rnnoise/)

Models `librnnoise_5h_en_500k` and `librnnoise_5h_b_500k` have **almost the same quality** of noise reduction. `librnnoise_5h_ru_500k` is most **suitable for working with Russian speech**, and `librnnoise_5h_b_500k` is **for mixed speech** or speech in a non-Russian language, it is more universal.

Comparative examples of how the new models work with the standard one are available in [`test_audio/comparative_tests`](https://github.com/Desklop/RNNoise_Wrapper/tree/master/test_audio/comparative_tests).

This wrapper on Intel i7-10510U CPU **works 28-30 times faster than real time** when denoising an entire audio recording, and **18-20 times faster than real time** when working in streaming mode (i.e. processing 20ms audio fragments). In this case, only 1 core was involved, the load on which was about 80-100%.

## Installation

This wrapper has the following dependencies: [pydub](https://github.com/jiaaro/pydub) and [numpy](https://github.com/numpy/numpy).

Installing with pip:

```bash
pip install git+https://github.com/Desklop/RNNoise_Wrapper
```

**WARNING!** Before using the wrapper, RNNoise must be compiled. If you are using **Linux or Mac**, you can use the **pre-compiled RNNoise** (on Ubuntu 19.10 64 bit) that **comes with the package** (it also works in Google Colaboratory). If the standard binary doesn't work for you, try manually compiling RNNoise. To do this, you must first prepare your OS (assuming `gcc` is already installed):

```bash
sudo apt-get install autoconf libtool
```

And execute:

```bash
git clone https://github.com/Desklop/RNNoise_Wrapper
cd RNNoise_Wrapper
./compile_rnnoise.sh
```

After that, the file `librnnoise_default.so.0.4.1` will appear in the `rnnoise_wrapper/libs` folder. The path to this binary file must be passed when creating an object of the RNNoise class from this wrapper (see below for details).

If you are using **Windows** then you need to **manually compile RNNoise**. The above instruction will not work, **use** these **links**: [one](https://github.com/xiph/rnnoise/issues/34), [two](https://github.com /jagger2048/rnnoise-windows). After compilation, the path to the binary file must be passed when creating an object of the RNNoise class from this wrapper (see below for details).

## Usage

### **one. In Python code**

**Suppress audio noise** `test.wav` and save the result as `test_denoised.wav`:

```python
from rnnoise_wrapper import RNNoise

denoiser = RNNoise()

audio = denoiser.read_wav('test.wav')
denoised_audio = denoiser.filter(audio)
denoiser.write_wav('test_denoised.wav', denoised_audio)
```

**Noise reduction in streaming audio** (buffer size is 20 milliseconds, i.e. 2 frames) (the example uses a stream simulation by processing the `test.wav` audio recording in parts, saving the result as `test_denoised_stream.wav`):

```python
audio = denoiser.read_wav('test.wav')

denoised_audio = b''
buffer_size_ms = 20

for i in range(buffer_size_ms, len(audio), buffer_size_ms):
    denoised_audio += denoiser.filter(audio[i-buffer_size_ms:i].raw_data, sample_rate=audio.frame_rate)
if len(audio) % buffer_size_ms != 0:
    denoised_audio += denoiser.filter(audio[len(audio)-(len(audio)%buffer_size_ms):].raw_data, sample_rate=audio.frame_rate)

denoiser.write_wav('test_denoised_stream.wav', denoised_audio, sample_rate=audio.frame_rate)
```

**More wrapper examples** can be found in [`rnnoise_wrapper_functional_tests.py`](https://github.com/Desklop/RNNoise_Wrapper/blob/master/rnnoise_wrapper_functional_tests.py) and [`rnnoise_wrapper_comparative_test.py`](https ://github.com/Desklop/RNNoise_Wrapper/blob/master/rnnoise_wrapper_comparative_test.py).

The [RNNoise] class(https://github.com/Desklop/RNNoise_Wrapper/blob/master/rnnoise_wrapper/rnnoise_wrapper.py#L29) contains the following methods:

- [`read_wav()`](https://github.com/Desklop/RNNoise_Wrapper/blob/master/rnnoise_wrapper/rnnoise_wrapper.py#L256): takes the name of the .wav audio recording, converts it to a supported format (16 bit, mono ) and returns a `pydub.AudioSegment` object with an audio recording
- [`write_wav()`](https://github.com/Desklop/RNNoise_Wrapper/blob/master/rnnoise_wrapper/rnnoise_wrapper.py#L277): accepts the .wav name of the audio recording, a `pydub.AudioSegment` object (or a byte string with audio data without wav headers) and saves the audio recording under the given name
- [`filter()`](https://github.com/Desklop/RNNoise_Wrapper/blob/master/rnnoise_wrapper/rnnoise_wrapper.py#L150): accepts a `pydub.AudioSegment` object (or a byte string of audio data without wav headers ), brings it to a sample rate of 48000 Hz, **splits the audio into frames** (10 milliseconds long), **cleans them of noise, and returns** a `pydub.AudioSegment` object (or a byte string without wav headers) while preserving original sample rate
- [`filter_frame()`](https://github.com/Desklop/RNNoise_Wrapper/blob/master/rnnoise_wrapper/rnnoise_wrapper.py#L128): clear only one frame (10ms long, 16bit, mono, 48000Hz ) from noise (directly accessing the binary file of the RNNoise library)

Detailed information about the supported arguments and the operation of each method is found in the comments in the source code of these methods.

**The default model is `librnnoise_5h_b_500k`**. When creating an object of the `RNNoise` class from a wrapper, using the `f_name_lib` argument, you can specify another model (RNNoise binary):

- **`librnnoise_5h_en_500k`** or **`librnnoise_default`** to use one of the complete models
- full/partial name/path to the compiled RNNoise binary file

```python
denoiser_def = RNNoise(f_name_lib='librnnoise_5h_ru_500k')
denoiser_new = RNNoise(f_name_lib='path/to/librnnoise.so.0.4.1')
```

**Features of the main `filter()` method:**

- for the highest quality work, you need an audio recording of at least 1 second in length, on which both voice and noise are present (moreover, noise should ideally be before and after the voice). Otherwise, the quality of noise reduction will be worse.
- if parts of one audio recording are transmitted (audio stream noise reduction), then their length must be at least `10` ms and a multiple of `10` (because the RNNoise library only supports frames with a length of `10` ms). This option does not affect the quality of noise reduction.
- if the last frame of the transferred audio recording is less than `10` ms (or the part of the audio is transferred less than `10` ms), then it is padded with zeros to the required size. Because of this, there may be a slight increase in the length of the final audio recording after noise reduction.
- the RNNoise library additionally returns for each frame the probability of having a voice in this frame (as a number from `0` to `1`) and using the `voice_prob_threshold` argument, you can filter the frames by this value. If the probability is lower than `voice_prob_threshold`, then the frame will be removed from the audio recording

### **2. As a command line tool**

```bash
python3 -m rnnoise_wrapper.cli -i input.wav -o output.wav
```

или

```bash
rnnoise_wrapper -i input.wav -o output.wav
```

Where:

- `input.wav` - name of source .wav audio
- `output.wav` - the name of the .wav audio file where the audio recording will be saved after denoising

## Education

Instructions for training RNNoise on your own data can be found in [`TRAINING.md`](https://github.com/Desklop/RNNoise_Wrapper/tree/master/TRAINING.md).

---

If you have any questions or want to collaborate, you can email me: vladsklim@gmail.com or on [LinkedIn](https://www.linkedin.com/in/vladklim/).
