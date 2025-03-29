from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import cv2

np.random.seed(0)

def temporal_consistency(bubble_centers):
    consistency_scores = []
    for t in range(len(bubble_centers) - 1):
        frame1 = bubble_centers[t]
        frame2 = bubble_centers[t + 1]

        if len(frame1) == len(frame2):
            distances = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(frame1, frame2)]
            consistency_scores.append(np.mean(distances))
        else:
            continue

    if consistency_scores:
        return np.mean(consistency_scores)
    else:
        print("No valid consistency scores calculated. Returning 0.")
        return 0

def optical_flow_similarity(prev_frame, next_frame, prev_centers, next_centers):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    similarities = []
    for (x1, y1), (x2, y2) in zip(prev_centers, next_centers):
        try:
            dx, dy = flow[int(y1), int(x1)]
            motion_vector = np.array([x2 - x1, y2 - y1])
            flow_vector = np.array([dx, dy])

            if np.linalg.norm(motion_vector) > 0 and np.linalg.norm(flow_vector) > 0:
                cos_sim = np.dot(motion_vector, flow_vector) / (np.linalg.norm(motion_vector) * np.linalg.norm(flow_vector))
                similarities.append(cos_sim)
        except Exception as e:
            print(f"Error calculating optical flow for centers ({x1}, {y1}) and ({x2}, {y2}): {e}")
            continue
    return np.mean(similarities) if similarities else None

def object_recall_watershed(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    frame_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(frame_colored, markers)
    frame_colored[markers == -1] = [0, 0, 255]
    return np.max(markers) - 1

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0])
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
  return(o)


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        print(f"Invalid bounding box: {bbox}")
        return np.array([0, 0, 0, 0]).reshape((4, 1))

    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x,score=None):
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  count = 0
  def __init__(self,bbox):
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000.
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01
    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    self.frame_count += 1
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
        i -= 1
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

def parse_args():
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

def analyze_tracking(pred_tracks):
    total_objects = 0
    total_tracks = 0
    track_durations = []
    prev_frame_objects = set()
    new_objects = 0
    lost_objects = 0

    for frame_index in sorted(pred_tracks.keys()):
        frame = pred_tracks[frame_index]
        current_frame_objects = set(frame)
        total_objects += len(frame)
        new_objects += len(current_frame_objects - prev_frame_objects)
        lost_objects += len(prev_frame_objects - current_frame_objects)
        for obj in frame:
            track_durations.append(frame_index + 1)
        prev_frame_objects = current_frame_objects
        total_tracks += len(frame)

    avg_objects_per_frame = total_objects / len(pred_tracks) if len(pred_tracks) > 0 else 0
    avg_track_duration = np.mean(track_durations) if len(track_durations) > 0 else 0
    new_object_frequency = new_objects / len(pred_tracks) if len(pred_tracks) > 0 else 0
    lost_object_frequency = lost_objects / len(pred_tracks) if len(pred_tracks) > 0 else 0

    return {
        "avg_objects_per_frame": avg_objects_per_frame,
        "avg_track_duration": avg_track_duration,
        "new_object_frequency": new_object_frequency,
        "lost_object_frequency": lost_object_frequency
    }

if __name__ == "__main__":
    frames_folder = '/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/Видео/frame_detection'
    detections_folder = '/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/Видео/детекции_пузыри_3/'
    output_folder = '/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/finish/'
    os.makedirs(output_folder, exist_ok=True)

    tracker = Sort()
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])

    bubble_centers = {}

    for frame_number, frame_file in enumerate(frame_files):
        detections_path = os.path.join(detections_folder, f"frame_{frame_number:05d}.txt")
        detections = []

        with open(detections_path, "r") as f:
            for line in f:
                det = list(map(float, line.strip().split(",")))
                x1, y1, width, height, conf = det[1:]
                x2, y2 = x1 + width, y1 + height
                detections.append([x1, y1, x2, y2, conf])

        detections = np.array(detections) if detections else np.empty((0, 5))
        tracked_objects = tracker.update(detections)

        bubble_centers[frame_number] = []
        if len(tracked_objects) > 0:
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = obj
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                bubble_centers[frame_number].append((center_x, center_y))
            with open(os.path.join(output_folder, f"tracked_{frame_number:05d}.txt"), "w") as f:
                for obj in tracked_objects:
                    f.write(",".join(map(str, obj)) + "\n")
        current_frame = cv2.imread(os.path.join(frames_folder, frame_file))
        if current_frame is None:
            print(f"Error: Could not load frame {frame_number}")
            continue

        watershed_bubble_count = object_recall_watershed(current_frame)
        if frame_number > 0:
            prev_frame = cv2.imread(os.path.join(frames_folder, frame_files[frame_number - 1]))

            if prev_frame is None:
                print(f"Error: Could not load previous frame {frame_files[frame_number - 1]}")
                continue

            temporal_score = temporal_consistency(bubble_centers)
            flow_score = optical_flow_similarity(prev_frame, current_frame,
                                                 bubble_centers[frame_number - 1],
                                                 bubble_centers[frame_number])

            print(f"Frame {frame_number}: Temporal Consistency = {temporal_score}, "
                  f"Optical Flow Similarity = {flow_score}, "
                  f"Watershed Bubble Count = {watershed_bubble_count}")

    analysis = analyze_tracking(bubble_centers)

    print(f"Среднее количество объектов на кадр: {analysis['avg_objects_per_frame']:.2f}")
    print(f"Средняя продолжительность треков: {analysis['avg_track_duration']:.2f} кадров")
    print(f"Частота появления новых объектов (FP): {analysis['new_object_frequency']:.2f}")
    print(f"Частота исчезновения объектов (FN): {analysis['lost_object_frequency']:.2f}")