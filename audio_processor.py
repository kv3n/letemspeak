import numpy as np
import tensorflow as tf


def process_audio(waveform):
    processed_waveform = np.copy(waveform)
    processed_waveform = processed_waveform.astype(dtype=np.float32)
    spectrogram = tf.signal.stft(processed_waveform, frame_length=400, frame_step=160, fft_length=512)

    real_spectrogram = spectrogram[..., None]
    real_spectrogram = tf.concat([tf.math.real(real_spectrogram), tf.math.imag(real_spectrogram)], axis=-1)

    return real_spectrogram


def build_audio(spectrogram):
    complex_mask = tf.dtypes.complex(spectrogram[:, :, 0], spectrogram[:, :, 1])
    reconstructed_audio = tf.signal.inverse_stft(complex_mask, frame_length=400, frame_step=160, fft_length=512)
    reconstructed_audio = tf.expand_dims(reconstructed_audio, axis=1)

    audio_wav = tf.audio.encode_wav(reconstructed_audio, sample_rate=16000)
    tf.io.write_file('output/test.wav', audio_wav)
