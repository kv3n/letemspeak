# Let Em Speak!
Audio Isolation of speakers from a debate of speakers.

## Getting started

### Prerequisites
Python 3.6.7 and download required dependencies from the requirements.txt.

### Installation
```
git clone https://github.com/kv3n/letemspeak.git
cd letemspeak
py -m venv venv  # create virtual environment
source venv/bin/activate
pip install --upgrade pip # make sure you have the latest version of pip
sudo apt-get -y install cmake  # Install cmake for dlib
pip install -r requirements.txt
```

### Download Dataset and Pre-Trained Facenet Model
We downloaded the training and test files from [here](https://looking-to-listen.github.io/avspeech/download.html) and put them in our google drive.

So from the project root directory run the following commands to download the training and test files as well as the pre-trained facenet weights.

```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sTHFmc9tl7Kxw-aUnEfZn-X00Eg-73Q5' -O 'avspeech_train.csv'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xbSBR-nq15VhvW5ufT4WnzYC2HciBzaE' -O 'avspeech_test.csv'
mkdir pre-trained-models
cd pre-trained-models
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=12YXsDiPSxpzXP4m2yGiWMXXTMnH8lzwg' -O 'facenet_weights.h5'
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
