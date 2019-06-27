import os
import itertools

import cv2
import dlib
import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip
from audio_processor import build_audio

from face_embedding import InceptionResNetV1


def find_speaker_id(x, y, speakers):
    distances = []
    for speaker_id, speaker in speakers.items():
        diff_x = x - speaker['x']
        diff_y = y - speaker['y']
        dist = diff_x ** 2 + diff_y ** 2
        distances.append((dist, speaker_id))

    if not distances:
        return -1

    return min(distances)[1]


def detect_face(frames):
    speakers = dict()

    for frame_idx, frame in enumerate(frames):
        detector = dlib.get_frontal_face_detector()
        speaker_detections = detector(frame, 1)

        print('#{} -> speaker count: {}'.format(frame_idx, len(speaker_detections)))
        speakers_in_frame = dict()
        if speaker_detections:
            for i, d in enumerate(speaker_detections):
                left_x = d.left()
                right_x = d.right() + 1
                top_y = d.top()
                bottom_y = d.bottom() + 1

                speaker = frame[top_y:bottom_y, left_x:right_x, :]
                speaker = cv2.resize(speaker, (160, 160))  # Because of Facenet input shape

                avg_x = (left_x * 0.5 + right_x * 0.5) / frame.shape[1]
                avg_y = (top_y * 0.5 + bottom_y * 0.5) / frame.shape[0]

                speakers_in_frame[i] = {
                    'frames': speaker,
                    'x': avg_x,
                    'y': avg_y
                }

        # Sync Speakers
        while len(speakers) < len(speakers_in_frame):
            speakers[len(speakers)] = {
                'frames': [np.zeros(shape=(160, 160, 3), dtype=np.uint8) for _ in range(frame_idx)],
                'x': -1.0,
                'y': -1.0
            }

        # Find true assignment for speakers.
        possible_assignments = itertools.product(speakers_in_frame.keys(), speakers.keys())

        def distance(active_speaker_key, speaker_key):
            diff_x = speakers_in_frame[active_speaker_key]['x'] - speakers[speaker_key]['x']
            diff_y = speakers_in_frame[active_speaker_key]['y'] - speakers[speaker_key]['y']
            return diff_x ** 2 + diff_y ** 2

        distances = [(distance(active_speaker_key, speaker_key), active_speaker_key, speaker_key)
                     for active_speaker_key, speaker_key in possible_assignments]

        distances.sort()

        assignment_from = set()
        assignment_to = set()
        for _, active_speaker_key, speaker_key in distances:
            if active_speaker_key in assignment_from or speaker_key in assignment_to:
                continue

            cur_x = speakers[speaker_key]['x']
            new_x = speakers_in_frame[active_speaker_key]['x']
            cur_y = speakers[speaker_key]['y']
            new_y = speakers_in_frame[active_speaker_key]['y']

            speakers[speaker_key]['frames'].append(speakers_in_frame[active_speaker_key]['frames'])
            speakers[speaker_key]['x'] = new_x if cur_x < 0.0 else cur_x
            speakers[speaker_key]['y'] = new_y if cur_y < 0.0 else cur_y

            assignment_from.add(active_speaker_key)
            assignment_to.add(speaker_key)


        # Fill in frames for unused speakers
        unused_speakers = set(speakers.keys()) - assignment_to
        for speaker_id in unused_speakers:
            speakers[speaker_id]['frames'].append(np.zeros(shape=(160, 160, 3), dtype=np.uint8))


        # Uncomment the following only for DEBUGGING
        cv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Input', cv_frame)
        for speaker_key, speaker in speakers.items():
            cv_speaker_frame = cv2.cvtColor(speaker['frames'][frame_idx], cv2.COLOR_RGB2BGR)
            cv2.imshow('Speaker #{} @ #{}'.format(speaker_key, frame_idx), cv_speaker_frame)
            print('Frame #{}, Speaker #{} at {}, {}'.format(frame_idx, speaker_key, speaker['x'], speaker['y']))
        cv2.waitKey()
        cv2.destroyAllWindows()

    return speakers


def fetch_embeddings(speakers):
    facenet = InceptionResNetV1(weights_path='pre-trained-models/facenet_weights.h5')
    get_output = tf.keras.backend.function([facenet.layers[0].input],
                                           [facenet.get_layer('Dropout').output])  # Should we use AvgPool instead?

    embeddings = []
    for speaker_frames in speakers:
        speaker_embeddings = []
        for frame in speaker_frames:
            if frame.any():
                in_speakers = np.array([frame])
                frame_embeddings = get_output(in_speakers)[0]
            else:
                frame_embeddings = np.zeros(shape=(1, 1792), dtype=np.float32)  # Because the avg pool from InceptionNet
            speaker_embeddings.append(frame_embeddings)

        embeddings.append(np.array(speaker_embeddings))

    return embeddings


def process_video(frames, ground_truth):
    detected_speakers = detect_face(frames)

    if ground_truth:
        true_speaker_id = find_speaker_id(ground_truth[0], ground_truth[1], detected_speakers)
        if true_speaker_id < 0:
            return None

        true_speakers = [detected_speakers[true_speaker_id]['frames']]
    else:
        true_speakers = []
        for speaker_id, speaker in detected_speakers.items():
            true_speakers.append(speaker['frames'])

    embeddings = fetch_embeddings(true_speakers)

    return embeddings


def save_val_results(key, start, results, output_dir='', data_dir='data/test'):

    if not os.path.exists('output/{}'.format(output_dir)):
        os.mkdir('output/{}'.format(output_dir))

    filename = '{}/{}.mp4'.format(data_dir, key)
    video = VideoFileClip(filename).set_fps(25)

    for idx, result in enumerate(results):
        result = np.squeeze(result, axis=0)

        # Extract clip and original audio
        sub_video = video.subclip(start, start + 3.0)
        original_audio = sub_video.audio.set_fps(16000).to_soundarray()[:, 0]

        # Replace Audio with masked audio
        sub_video.audio = build_audio(result, original_audio)
        sub_video.write_videofile('output/{}/{}_{}_{}.mp4'.format(output_dir, key, int(round(start)), idx),
                                  codec="libx264", audio_codec="aac")
