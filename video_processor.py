import numpy as np
import cv2
import dlib


def detect_face(frame_idx, frame):
    detector = dlib.get_frontal_face_detector()
    speaker_detections = detector(frame, 1)
    speakers = []
    for i, d in enumerate(speaker_detections):
        speaker = frame[d.top():d.bottom(), d.left():d.right(), :]
        center_x = (d.left() + d.right()) * 0.5 / frame.shape[1]
        center_y = (d.top() + d.bottom()) * 0.5 / frame.shape[0]

        speakers.append((speaker, center_x, center_y))

    # Uncomment the following only for DEBUGGING
    # print('#{} -> speaker count: {}'.format(frame_idx, len(speakers)))
    # cv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow('Input', cv_frame)
    # for speaker_idx, speaker in enumerate(speakers):
        # cv_speaker_frame = cv2.cvtColor(speaker[0], cv2.COLOR_RGB2BGR)
        # cv2.imshow('Speaker #{}'.format(speaker_idx+1), cv_speaker_frame)
        # print('Frame #{}, Speaker #{} at {}, {}'.format(frame_idx, speaker_idx+1, speaker[1], speaker[2]))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return speakers


def process_video(frames, ground_truth):
    detected_speakers = [detect_face(idx, frame) for idx, frame in enumerate(frames)]

    if ground_truth:
        true_speakers = []
        for frame_id, frame in enumerate(detected_speakers):
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
            true_speakers.append([frame[closest_speaker[1]][0]])
    else:
        true_speakers = []
        for frame_id, frame in enumerate(detected_speakers):
            speakers_in_frame = [speaker[0] for speaker_idx, speaker in enumerate(frame)]

            true_speakers.append(speakers_in_frame)

    return true_speakers
