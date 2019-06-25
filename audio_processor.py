import numpy as np
import tensorflow as tf
from moviepy.editor import AudioFileClip


def get_spectrogram(waveform):
    processed_waveform = np.copy(waveform)
    processed_waveform = processed_waveform.astype(dtype=np.float32)
    spectrogram = tf.signal.stft(processed_waveform, frame_length=400, frame_step=160, fft_length=512)

    return spectrogram


def break_complex_spectrogram(spectrogram):
    real_spectrogram = spectrogram[..., None]
    real_spectrogram = tf.concat([tf.math.real(real_spectrogram), tf.math.imag(real_spectrogram)], axis=-1)

    return real_spectrogram


def process_audio(waveform):
    spectrogram = get_spectrogram(waveform)

    return break_complex_spectrogram(spectrogram)


def is_complex(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor.dtype == np.complex64
    else:
        return tensor.dtype == tf.complex64


def apply_mask(spectrogram, mask):
    if not is_complex(spectrogram):
        spectrogram = tf.dtypes.complex(spectrogram[:, :, 0], spectrogram[:, :, 1])

    if not is_complex(mask):
        mask = tf.dtypes.complex(mask[:, :, 0], mask[:, :, 1])

    masked_audio = tf.math.multiply(spectrogram, mask)

    return masked_audio


def build_audio(spectrogram, original_audio):
    spectrogram_reconstructed = apply_mask(get_spectrogram(original_audio), spectrogram)
    reconstructed_audio = tf.signal.inverse_stft(spectrogram_reconstructed,
                                                 frame_length=400, frame_step=160, fft_length=512)
    reconstructed_audio = tf.expand_dims(reconstructed_audio, axis=1)

    audio_wav = tf.audio.encode_wav(reconstructed_audio, sample_rate=16000)
    tf.io.write_file('output/temp.wav', audio_wav)
    audio = AudioFileClip('output/temp.wav')

    return audio
