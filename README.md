# Let Em Speak!
Audio Isolation of speakers from a debate of speakers.

## Getting started

### Prerequisites
Python 3.6.7 and download required dependencies from the requirements.txt.

### Download Dataset
Download train and test files from [here](https://looking-to-listen.github.io/avspeech/download.html) and place them in the project root directory.

### Installation
```
git clone https://github.com/kv3n/letemspeak.git
cd letemspeak
py -m venv venv  # create virtual environment
source venv/bin/activate
pip install --upgrade pip # make sure you have the latest version of pip
pip install -r requirements.txt
```

### Usage
To prep the data, run
```
python data_utils.py
```

## Authors
- [Kishore Venkateshan](https://github.com/kv3n)
- [Hsuan-Hau Liu](https://github.com/hsuanhauliu)

## References
- [Looking to Listen: Audio-Visual Speech Separation](https://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html)
