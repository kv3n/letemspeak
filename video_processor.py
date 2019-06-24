import cv2
import dlib

import numpy as np
import tensorflow as tf
from face_embedding import InceptionResNetV1

from moviepy.editor import VideoFileClip
from audio_processor import build_audio


def detect_face(frame_idx, frame):
    detector = dlib.get_frontal_face_detector()
    speaker_detections = detector(frame, 1)
    speakers = []
    for i, d in enumerate(speaker_detections):
        left_x = d.left()  # max(d.left() - 20, 0)
        right_x = d.right() + 1  # min(d.right() + 20, frame.shape[1])
        top_y = d.top()  # max(d.top() - 20, 0)
        bottom_y = d.bottom() + 1  # min(d.bottom() + 20, frame.shape[0])

        speaker = frame[top_y:bottom_y, left_x:right_x, :]
        speaker = cv2.resize(speaker, (160, 160))  # Because of Facenet input shape

        speakers.append((speaker, left_x, right_x, top_y, bottom_y))

    # Uncomment the following only for DEBUGGING
    # print('#{} -> speaker count: {}'.format(frame_idx, len(speakers)))
    # cv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow('Input', cv_frame)
    # for speaker_idx, speaker in enumerate(speakers):
    #     cv_speaker_frame = cv2.cvtColor(speaker[0], cv2.COLOR_RGB2BGR)
    #     cv2.imshow('Speaker #{}'.format(speaker_idx+1), cv_speaker_frame)
    #     print('Frame #{}, Speaker #{} at {}, {}'.format(frame_idx, speaker_idx+1, speaker[1], speaker[2]))
    # if frame_idx == 57:
    #     cv2.waitKey()
    # else:
    #     cv2.waitKey(5)
    # cv2.destroyAllWindows()

    return speakers


def clean_frames(speaker_frames, frames):
    cleaned_frames = []

    for frame_idx, speaker_frame in enumerate(speaker_frames):
        cleaned_speaker = []

        """
        # The paper claims that this is not needed
        if not speaker_frame:
            if frame_idx == 0:
                print('ERROR: DISCARD THIS CLIP')

            for prev_speaker in speaker_frames[frame_idx-1]:
                left_x = prev_speaker[1]
                right_x = prev_speaker[2]
                top_y = prev_speaker[3]
                bottom_y = prev_speaker[4]

                speaker = frames[frame_idx][top_y:bottom_y, left_x:right_x, :]
                speaker = cv2.resize(speaker, (160, 160))  # Because of Facenet input shape

                speaker_frame.append((speaker, left_x, right_x, top_y, bottom_y))
        """

        for speaker in speaker_frame:
            center_x = (speaker[1] + speaker[2]) * 0.5 / frames[frame_idx].shape[1]
            center_y = (speaker[3] + speaker[4]) * 0.5 / frames[frame_idx].shape[0]

            cleaned_speaker.append((speaker[0], center_x, center_y))

        cleaned_frames.append(cleaned_speaker)

    return cleaned_frames


def fetch_embeddings(frames):
    facenet = InceptionResNetV1(weights_path='pre-trained-models/facenet_weights.h5')
    get_output = tf.keras.backend.function([facenet.layers[0].input],
                                           [facenet.get_layer('Dropout').output])  # Should we take from AvgPool instead?

    embeddings = []
    for frame in frames:
        if frame:
            in_speakers = np.array(frame)
            frame_embeddings = get_output(in_speakers)[0]
        else:
            frame_embeddings = np.zeros(shape=(1, 1792), dtype=np.float32) # Because the avg pool from InceptionNet

        embeddings.append(frame_embeddings)

    return np.array(embeddings)


def process_video(frames, ground_truth):
    detected_speakers = [detect_face(idx, frame) for idx, frame in enumerate(frames)]
    cleaned_frames = clean_frames(detected_speakers, frames)

    if ground_truth:
        true_speakers = []
        for frame_id, frame in enumerate(cleaned_frames):
            closest_speaker = (float('inf'), -1)
            for speaker_idx, speaker in enumerate(frame):
                x_distance = ground_truth[0] - speaker[1]
                y_distance = ground_truth[1] - speaker[2]
                distance_to_ground_truth = x_distance ** 2 + y_distance ** 2
                closest_speaker = min(closest_speaker, (distance_to_ground_truth, speaker_idx))

            # print('frame #{} -> true speaker at {}, {} vs gt at {}, {}'.format(frame_id+1,
            #                                                                    frame[closest_speaker[1]][1],
            #                                                                    frame[closest_speaker[1]][2],
            #                                                                    ground_truth[0],
            #                                                                    ground_truth[1]))
            if closest_speaker[1] >= 0:
                true_speakers.append([frame[closest_speaker[1]][0]])
            else:
                true_speakers.append([])
    else:
        true_speakers = []
        for frame_id, frame in enumerate(cleaned_frames):
            speakers_in_frame = [speaker[0] for speaker_idx, speaker in enumerate(frame)]

            true_speakers.append(speakers_in_frame)

    true_embeddings = fetch_embeddings(true_speakers)

    return true_embeddings


def save_val_results(key, start, results, data_dir='data/test'):
    filename = '{}/{}.mp4'.format(data_dir, key)
    video = VideoFileClip(filename).set_fps(25)

    for idx, result in enumerate(results):
        sub_video = video.subclip(start, start + 3.0)
        audio = build_audio(result)
        sub_video.audio = audio
        sub_video.write_videofile('output/{}_{}_{}.mp4'.format(key, int(round(start)), idx),
                                  codec="libx264", audio_codec="aac")
