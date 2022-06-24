# RNNoise training

The dataset from [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge) was used for training. The training was performed on a machine with an Intel Core i7-10510U CPU and 16Gb RAM, Ubuntu 19.10 64bit OS.

**The instructions assume that you have already cloned the repositories** from the Microsoft DNS Challenge and the current RNNoise_Wrapper repository.

### **one. Preparing the Microsoft DNS Challenge dataset**

**Preparing data in Microsoft DNS Challenge** by updating `noisyspeech_synthesizer.cfg` and running `python3 noisyspeech_synthesizer_singleprocess.py` as per [original instruction in repository](https://github.com/microsoft/DNS-Challenge#usage ).

**Note 1.1. Scripts in DNS Challenge need simple edits:**

- replace backslash with normal in all paths in `noisyspeech_synthesizer.cfg` and `noisyspeech_synthesizer_singleprocess.py`
- adding the output of the currently processed audio `print('Processing file #{}...'.format(file_num))` after line 165 in [`noisyspeech_synthesizer_singleprocess.py`](https://github.com/microsoft/DNS- challenge/blob/master/noisyspeech_synthesizer_singleprocess.py#L165) (for convenience, the data preparation process is rather slow)
- adding `return np.zeros(0), 16000` to exception handling in `audioread()` after line 43 in [`audiolib.py`](https://github.com/microsoft/DNS-Challenge/blob/ master/audiolib.py#L43)

**Note 1.2.** By default, only one folder with pure speech can be selected in the config, i.e. only [one of the available languages](https://github.com/microsoft/DNS-Challenge/tree/master/datasets/clean). **In order to use all available audio with pure speech (about 750 hours), you need to move the folders with speech in other languages ​​to the folder with English speech or to a new separate folder.**

Moving folders with speech in other languages ​​to the folder with English speech (perform in `DNS-Challenge/datasets/clean`):

```bash
mv -v french_data german_speech italian_speech russian_speech spanish_speech read_speech
```

Revert everything back:

```bash
mv -v read_speech/french_data read_speech/german_speech read_speech/italian_speech read_speech/russian_speech read_speech/spanish_speech ../clean
```

**Note 1.3.** If you need to train not on all available data (but only, for example, on 5 hours of them), but using all available languages, **it is recommended to pre-cut the number of audio recordings for each language by the smallest of them** (the least data for the Russian language, about 47 hours). That is, **balance languages ​​by the number of audio recordings** in them.

This can be done with the script [`training_utils/balance_dns_challenge_dataset.py`](https://github.com/Desklop/RNNoise_Wrapper/blob/master/training_utils/balance_dns_challenge_dataset.py) (it is better to copy the script to the dataset folder, run it in `DNS- challenge`):

```bash
python3 balance_dns_challenge_dataset.py -rf datasets/clean -sf russian_speech,read_speech,french_data,german_speech,italian_speech,spanish_speech -bsf all_balanced_speech
```

The balanced pure speech will be stored in the `all_balanced_speech` folder. The original structure of the audio recordings is broken (all audio recordings will be inside the specified folder, without subfolders), and the original names of the audio recordings are preserved.

**IMPORTANT!** Note 2 and Note 3 are **mutually exclusive**. To avoid problems, **recommended to use note 3 only.**

**Prepared dataset consists of 3 folders:**

- `clean` - audio recordings with clean speech, each 30 seconds long
- `noise` - audio recordings with noise, each 30 seconds long
- `noisy` - audio recordings with noisy speech (superimposed noise from `noise` on speech from `clean`), each also 30 seconds long

RNNoise training requires only the `clean` and `noise` folders. **It is recommended to copy them to `RNNoise_Wrapper/datasets`.** For convenience, the dataset has been given the **name `test_training_set`**, i.e. the full path to the dataset will be `RNNoise_Wrapper/datasets/test_training_set`.

### **2. Preparing the environment for RNNoise**

Before working with RNNoise, you need to either clone the original [repository] (https://github.com/xiph/rnnoise) or extract a copy of it in the current project:

```bash
unzip rnnoise_master_11/20/2020.zip
```

To compile RNNoise and its tools, you must first prepare the OS (assuming `gcc` is already installed):

```bash
sudo apt-get install autoconf libtool
```

Then build the RNNoise tools (execute in `RNNoise_Wrapper`):

```bash
cd rnnoise-master/src && ./compile.sh && cd -
```

And prepare the Python virtual environment for learning:

```bash
virtualenv --python=python3.7 env_train
source env_train/bin/activate
pip install -r requirements_train.txt
```

### **3. Combining audio recordings in a dataset**

**RNNoise training requires 2 audio recordings: pure speech and noisy.** Audio recordings must be in .raw format, mono, 16 bits and 48000 Hz. That is, for training, it is necessary to combine all the audio recordings in the dataset into 2 large ones.

You can merge and prepare all audio recordings with pure speech and all audio recordings with noise with the script [`training_utils/prepare_dataset_for_training.py`](https://github.com/Desklop/RNNoise_Wrapper/blob/master/training_utils/prepare_dataset_for_training.py) (run in `RNNoise_Wrapper`):

```bash
python3 training_utils/prepare_dataset_for_training.py -cf datasets/test_training_set/clean -nf datasets/test_training_set/noise -bca datasets/test_training_set/all_clean.raw -bna datasets/test_training_set/all_noise.raw
```

### **four. Formation of the training sample**

After combining the audio recordings, ** it is necessary to extract the coefficients from them and form a ready-made training sample. ** The key parameter is the size of the data matrix. It defaults to `500000x87`. **It is recommended to change the first dimension depending on the size of the dataset.**

**Note 4.1.** I have tried training with `500000`, `1000000` and `5000000`. I recommend trying a step less: `500000`, `1000000`, `2000000`. Or even `500000`, `1000000`, `1500000`, `2000000`. So it will be easier then to select the most successful / high-quality model.

Formation of a training sample (perform in `RNNoise_Wrapper`):

```bash
rnnoise-master/src/denoise_training datasets/test_training_set/all_clean.raw datasets/test_training_set/all_noise.raw 5000000 > train_logs/test_training_set/training_test_b_500k.f32
```

Converting the training sample to `.h5` (perform in `RNNoise_Wrapper`):

```bash
python3 rnnoise-master/training/bin2hdf5.py train_logs/test_training_set/training_test_b_500k.f32 5000000 87 train_logs/test_training_set/training_test_b_500k.h5
```

### **5. Model training**

Before running the training, you need to **copy the updated script** from [`training_utils/rnn_train_mod.py`](https://github.com/Desklop/RNNoise_Wrapper/blob/master/training_utils/rnn_train_mod.py) to `rnnoise-master/ training`.

The updated training script **differs** from the original one in **support for command line arguments and improved logs**.

Start training (perform in `RNNoise_Wrapper`):

```bash
python3 rnnoise-master/training/rnn_train_mod.py train_logs/test_training_set/training_test_b_500k.h5 train_logs/test_training_set/weights_test_b_500k.hdf5
```

Training lasts **120 epochs**. After training is completed, **the weights of the resulting model are stored in `train_logs/test_training_set/weights_test_b_500k.hdf5`**. On the previously mentioned machine, training took about 45 minutes.

**Note 5.1.** To run GPU training, you need to install `tensorflow-gpu==1.15.4`. On the NVIDIA RTX2080Ti, the learning process used about 10GB of VRAM.

### **6. Model conversion**

RNNoise is written in C, so the resulting trained tensorflow **model needs to be converted to C** code.

The source converter in the project repository causes errors when trying to compile RNNoise with a new model. **Corrected converter needs to be copied** from [`training_utils/dump_rnn_mod.py`](https://github.com/Desklop/RNNoise_Wrapper/blob/master/training_utils/dump_rnn_mod.py) to `rnnoise-master/training`. The fixes are based on [issue in RNNoise source repository](https://github.com/xiph/rnnoise/issues/74#issuecomment-517075991).

Model conversion (perform in `RNNoise_Wrapper`):

```bash
python3 rnnoise-master/training/dump_rnn_mod.py train_logs/test_training_set/weights_test_b_500k.hdf5 rnnoise-master/src/rnn_data.c rnnoise-master/src/rnn_data.h
```

**Note 6.1.** Changing the names and locations of the final `.c` and `.h` files is not recommended. Otherwise, you will need to modify scripts to compile RNNoise.

### **7. Building RNNoise with the new model**

To test and use the new model **need to compile RNNoise** with updated `src/rnn_data.c` and `src/rnn_data.h` (run in `RNNoise_Wrapper`):

```bash
cd rnnoise-master && make clean && ./autogen.sh && ./configure && make && cd -
```

After a successful build, for convenience, you can copy the resulting binary to the folder with the trained model and its weights:

```bash
cp rnnoise-master/.libs/librnnoise.so.0.4.1 train_logs/test_training_set/librnnoise_test_b_500k.so.0.4.1
```

### **eight. Testing a new model**

**To evaluate the performance** of the resulting model, it is recommended to **run a comparison test with the standard model** using [`rnnoise_wrapper_comparative_test.py`](https://github.com/Desklop/RNNoise_Wrapper/blob/master/rnnoise_wrapper_comparative_test.py ).

## For reference

**The repository also contains [`Dockerfiles`](https://github.com/Desklop/RNNoise_Wrapper/tree/master/training_utils) for running training in a docker container**, both on CPU and GPU. This **may be useful on older CPUs** that don't support AVX instructions and won't install TensorFlow from pip (in this case, you can look for a suitable tensorflow package in [`optimized_tensorflow_wheels`](https://github.com/Desklop /optimized_tensorflow_wheels) or Google).

**List of useful bash commands:**

1. Copy folders with contents: `cp -R source source_copy`
2. Move folders with contents: `mv -v source_1 source_2 source_N all_source`
3. Copy 10 random files from the current folder: `ls | shuf -n 10 | xargs -i cp {} ../random_audio`
4. Calculate folder size: `du -hs datasets/clean`
5. Counting the number of files in the current folder and all its subfolders: `ls -laR | grep "^-" | wc`
6. Archive the folder in .zip: `zip -r test_training_set.zip datasets/test_training_set`
7. Unzip the archive to the current folder: `unzip test_training_set.zip`

## Sources

Documentation and issues on which this instruction is based:

1. https://github.com/xiph/rnnoise/blob/master/TRAINING-README
2. https://github.com/xiph/rnnoise/issues/8#issuecomment-346947946
3. https://github.com/xiph/rnnoise/issues/1#issuecomment-467170166
4. https://github.com/xiph/rnnoise/issues/74

---

If you have any questions or want to collaborate, you can email me: vladsklim@gmail.com or on [LinkedIn](https://www.linkedin.com/in/vladklim/).
