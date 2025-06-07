import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt
import time
import torch
import sqlite3
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from CounTR import models_mae_cross
from PIL import Image
from torchvision import transforms

# Константы
max_disappeared = 3
nms_radius = 12
density_threshold = 0.05
max_positions_history = 5
iou_dist_weights = (0.4, 0.6)
min_object_size = 5
min_distance = 15
max_cost_threshold = 0.6
border_disappear_multiplier = 2
shot_num = 0
model_size = 384
bbox_size = 10

class DeepSortTracker:
    def __init__(self, img_size, nms_max_overlap=0.6, max_cosine_distance=0.5, nn_budget=None,
                 max_age=30, min_hits=3, iou_threshold=0.3):
        self.img_size = img_size
        self.nms_max_overlap = nms_max_overlap
        self.iou_threshold = iou_threshold
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age=max_age, n_init=min_hits)

    def prepare_detections(self, yolo_detections):
        detections = []
        for det in yolo_detections:
            x1, y1, x2, y2, conf, cl = det
            bbox = (x1, y1, x2 - x1, y2 - y1)  # Конвертация в формат (x,y,w,h)
            feature = []
            detections.append(Detection(bbox, conf, feature))
        return detections

    def update(self, yolo_detections):
        detections = self.prepare_detections(yolo_detections)
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(detections)

        results = []
        for track in self.tracker.tracks:
            print("track", track, track.is_confirmed(), track.time_since_update)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return results


class TrackingQualityAnalyzer:
    def __init__(self):
        self.metrics = {
            'frame': [],
            'displacement': [],
            'coverage': [],
            'optical_flow': [],
            'temporal_consistency': [],
            'track_lengths': {},
            'active_tracks': []
        }
        self.prev_tracks = {}
        self.prev_frame = None

    @staticmethod
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def update_metrics(self, frame_num, current_tracks, current_frame):
        current_dict = {}
        for track in current_tracks:
            if len(track) >= 5:  # Проверяем, что трек содержит все необходимые элементы
                track_id = track[0]
                bbox = track[1:5]  # x1, y1, x2, y2
                current_dict[track_id] = bbox

        for track_id in current_dict:
            self.metrics['track_lengths'][track_id] = self.metrics['track_lengths'].get(track_id, 0) + 1
        
        if not self.prev_tracks:
            self.prev_tracks = current_dict
            self.prev_frame = current_frame.copy()
            return

        matched = 0
        total_displacement = 0
        total_iou = 0
        total_flow = 0

        if self.prev_frame is not None:
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        for track_id, current_bbox in current_dict.items():
            if track_id in self.prev_tracks:
                matched += 1
                prev_bbox = self.prev_tracks[track_id]

                # Среднее смещение
                dx = current_bbox[0] - prev_bbox[0]
                dy = current_bbox[1] - prev_bbox[1]
                displacement = np.sqrt(dx**2 + dy**2)
                total_displacement += displacement

                # IoU для темпоральной согласованности
                iou = self.calculate_iou(prev_bbox, current_bbox)
                total_iou += iou

                # Анализ оптического потока в области объекта
                if self.prev_frame is not None:
                    x, y, w, h = map(int, prev_bbox)
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, current_frame.shape[1] - x), min(h, current_frame.shape[0] - y)
                    if w > 0 and h > 0:
                        obj_flow = flow[y:y+h, x:x+w]
                        if obj_flow.size > 0:
                            magnitude = np.sqrt(obj_flow[...,0]**2 + obj_flow[...,1]**2)
                            total_flow += np.mean(magnitude)

        # Полнота обнаружения
        coverage = matched / len(self.prev_tracks) if len(self.prev_tracks) > 0 else 0

        self.metrics['frame'].append(frame_num)
        self.metrics['displacement'].append(total_displacement/matched if matched > 0 else 0)
        self.metrics['coverage'].append(coverage)
        self.metrics['temporal_consistency'].append(total_iou/matched if matched > 0 else 0)
        self.metrics['optical_flow'].append(total_flow/matched if matched > 0 else 0)
        self.metrics['active_tracks'].append(len(current_dict))

        self.prev_tracks = current_dict
        self.prev_frame = current_frame.copy()

    def get_final_metrics(self):
        if not self.metrics['frame']:
            return {}

        avg_metrics = {
            'avg_displacement': np.mean(self.metrics['displacement']),
            'avg_coverage': np.mean(self.metrics['coverage']),
            'avg_temporal_consistency': np.mean(self.metrics['temporal_consistency']),
            'avg_optical_flow': np.mean(self.metrics['optical_flow']),
            'track_length_mean': np.mean(list(self.metrics['track_lengths'].values())) if self.metrics['track_lengths'] else 0,
            'track_length_median': np.median(list(self.metrics['track_lengths'].values())) if self.metrics['track_lengths'] else 0,
            'max_active_tracks': max(self.metrics['active_tracks']) if self.metrics['active_tracks'] else 0
        }
        return avg_metrics

    def get_tracking_score(self, weights=None, normalize=True, reference_score=22.5190):
        final_metrics = self.get_final_metrics()
        print("New metrics keys:", final_metrics.keys())
        if not final_metrics:
            print("Предупреждение: нет метрик для расчета score!")
            return 0.0
        print(final_metrics)
        default_weights = {
            'avg_displacement': -0.2,
            'avg_coverage': 0.35,
            'avg_temporal_consistency': 0.25,
            'avg_optical_flow': -0.1,
            'track_length_mean': 0.2,
            'max_active_tracks': 0.1
        }

        weights = weights if weights is not None else default_weights

        missing_metrics = [k for k in weights if k not in final_metrics]
        
        if missing_metrics:
            print(f"Предупреждение: отсутствуют метрики {missing_metrics}, они не учитываются в score")

        raw_score = 0.0
        for key, weight in weights.items():
            if key in final_metrics:
                raw_score += float(final_metrics[key]) * weight

        if normalize:
            if reference_score <= 0:
                print("Ошибка: reference_score должен быть положительным!")
                return 0.0

            normalized_score = (raw_score / reference_score) - 1
            
            return (expit(normalized_score)) * 2  # Применяем сигмоиду для получения значения в диапазоне (0, 1)
        else:
            return raw_score
        
    def generate_metrics_plots(self, save_path=None):
        if not self.metrics['frame']:
            return None
        
        plt.figure(figsize=(16, 12))

        plt.subplot(3, 2, 1)
        plt.plot(self.metrics['frame'], self.metrics['displacement'], color='blue')
        plt.title('Динамика смещения объектов')
        plt.grid(True)

        plt.subplot(3, 2, 2)
        plt.plot(self.metrics['frame'], self.metrics['coverage'], color='green')
        plt.title('Полнота трекинга')
        plt.grid(True)

        plt.subplot(3, 2, 3)
        plt.plot(self.metrics['frame'], self.metrics['displacement'])
        plt.plot(self.metrics['frame'], self.metrics['coverage'])
        plt.title('Качество трекинга')
        plt.grid(True)

        plt.subplot(3, 2, 4)
        plt.plot(self.metrics['frame'], self.metrics['temporal_consistency'], color='red')
        plt.title('Темпоральная согласованность')
        plt.grid(True)

        plt.subplot(3, 2, 5)
        plt.plot(self.metrics['frame'], self.metrics['optical_flow'], color='purple')
        plt.title('Оптический поток объектов')
        plt.grid(True)

        plt.subplot(3, 2, 6)
        plt.plot(self.metrics['frame'], self.metrics['active_tracks'], color='green')
        plt.title('Активные треки')
        plt.grid(True)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return plt.gcf()
        
    def plot_metrics(self, save_path=None):
        fig = self.generate_metrics_plots(save_path)
        if fig is None:
            st.warning("Нет данных для построения графиков!")
            return
        
        st.subheader("Графики метрик")
        st.pyplot(fig)
        plt.close(fig)

class TrackingDatabase:
    def __init__(self, db_name='tracking_analysis.db'):
        self.db_name = db_name
        self.init_db()
        
    def init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            # Создаем таблицу с правильными типами данных
            c.execute('''CREATE TABLE IF NOT EXISTS analyses
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          filename TEXT,
                          model_name TEXT,
                          avg_displacement REAL,
                          tracking_coverage REAL,
                          temporal_consistency REAL,
                          optical_flow REAL,
                          avg_track_length REAL,
                          max_active_tracks INTEGER,
                          tracking_score REAL,
                          processing_time REAL,
                          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit()
    
    def save_analysis(self, data):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            # Убедимся, что все значения имеют правильный тип
            try:
                c.execute('''INSERT INTO analyses 
                             (filename, model_name, avg_displacement, tracking_coverage, 
                              temporal_consistency, optical_flow, 
                              avg_track_length, max_active_tracks, tracking_score, processing_time)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                          (
                              str(data.get('filename', '')),
                              str(data.get('model_name', '')),
                              float(data.get('avg_displacement', 0)),
                              float(data.get('tracking_coverage', 0)),
                              float(data.get('temporal_consistency', 0)),
                              float(data.get('optical_flow', 0)),
                              float(data.get('avg_track_length', 0)),
                              int(data.get('max_active_tracks', 0)),
                              float(data.get('tracking_score', 0)),
                              float(data.get('processing_time', 0))
                          ))
                conn.commit()
            except Exception as e:
                st.error(f"Ошибка при сохранении в базу данных: {str(e)}")
                conn.rollback()
    
    def get_recent_analyses(self, limit=10):
        with sqlite3.connect(self.db_name) as conn:
            try:
                # Используем параметризованный запрос
                query = '''SELECT 
                            id,
                            filename,
                            model_name,
                            avg_displacement,
                            tracking_coverage,
                            temporal_consistency,
                            optical_flow,
                            avg_track_length,
                            max_active_tracks,
                            tracking_score,
                            processing_time,
                            timestamp
                          FROM analyses 
                          ORDER BY timestamp DESC 
                          LIMIT ?'''
                df = pd.read_sql(query, conn, params=(limit,))
                
                # Преобразуем числовые столбцы
                numeric_cols = ['avg_displacement', 'tracking_coverage', 'temporal_consistency',
                              'optical_flow', 'avg_track_length', 'max_active_tracks',
                              'tracking_score', 'processing_time']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            except Exception as e:
                st.error(f"Ошибка при чтении из базы данных: {str(e)}")
                return pd.DataFrame()
    
    def display_recent_analyses(self):
        df = self.get_recent_analyses()
        
        if df.empty:
            st.warning("В базе данных пока нет записей")
            return
        
        # Создаем копию для отображения
        display_df = df.drop(columns=['id']).copy()
        
        # Переименовываем столбцы для красивого отображения
        display_df = display_df.rename(columns={
            'filename': 'Файл',
            'model_name': 'Модель',
            'avg_displacement': 'Ср. смещение',
            'tracking_coverage': 'Покрытие',
            'temporal_consistency': 'Согласованность',
            'optical_flow': 'Опт. поток',
            'avg_track_length': 'Длина трека',
            'max_active_tracks': 'Макс. треков',
            'tracking_score': 'Оценка',
            'processing_time': 'Время (сек)',
            'timestamp': 'Время анализа'
        })
        
        # Форматируем данные
        styled_df = display_df.style.format({
            'Ср. смещение': "{:.2f}",
            'Покрытие': "{:.2%}",
            'Согласованность': "{:.2f}",
            'Опт. поток': "{:.2f}",
            'Длина трека': "{:.1f}",
            'Оценка': "{:.3f}",
            'Время (сек)': "{:.2f}"
        })
        
        # Отображаем таблицу
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=min(400, 35 * (len(display_df) + 35))
        )

class ObjectTracker:
    def __init__(self, max_disappeared=1, frame_size=(1920, 1080)):
        self.next_id = 1
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.frame_width, self.frame_height = frame_size
        self.metrics = TrackingQualityAnalyzer()
    
    def update(self, detections, current_frame=None):
        for obj_id in self.objects:
            if len(self.objects[obj_id]['positions']) > 1:
                last_pos = self.objects[obj_id]['positions'][-1]
                prev_pos = self.objects[obj_id]['positions'][-2]
                velocity = last_pos - prev_pos
                predicted_pos = last_pos + velocity
                self.objects[obj_id]['predicted_pos'] = predicted_pos
            else:
                self.objects[obj_id]['predicted_pos'] = self.get_center(self.objects[obj_id]['bbox'])
        
        for obj_id in list(self.objects.keys()):
            bbox = self.objects[obj_id]['bbox']
            if (bbox[0] <= 5 or bbox[1] <= 5 or 
                bbox[2] >= self.frame_width - 5 or bbox[3] >= self.frame_height - 5):
                self.objects[obj_id]['disappeared'] += border_disappear_multiplier
            else:
                self.objects[obj_id]['disappeared'] += 1
                
            if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                del self.objects[obj_id]
        
        if len(detections) == 0:
            return self._prepare_output(current_frame)
        
        if len(self.objects) == 0:
            for det in detections:
                det_center = self.get_center(det)
                self.objects[self.next_id] = {
                    'bbox': det,
                    'disappeared': 0,
                    'positions': [det_center],
                    'predicted_pos': det_center
                }
                self.next_id += 1
            return self._prepare_output(current_frame)
        
        obj_ids = list(self.objects.keys())
        obj_bboxes = [self.objects[obj_id]['bbox'] for obj_id in obj_ids]
        
        cost_matrix = np.zeros((len(obj_bboxes), len(detections)))
        for i, obj_bbox in enumerate(obj_bboxes):
            obj_predicted = self.objects[obj_ids[i]]['predicted_pos']
            for j, det_bbox in enumerate(detections):
                det_center = self.get_center(det_bbox)
                iou_score = 1 - self.calculate_iou(obj_bbox, det_bbox)
                dist_score = np.linalg.norm(obj_predicted - det_center) / 100
                cost_matrix[i, j] = iou_dist_weights[0] * iou_score + iou_dist_weights[1] * dist_score
        
        matched_obj_indices = set()
        matched_det_indices = set()
        
        while True:
            min_cost = np.min(cost_matrix)
            if min_cost > max_cost_threshold:
                break
                
            i, j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            obj_id = obj_ids[i]
            
            self.objects[obj_id]['bbox'] = detections[j]
            self.objects[obj_id]['disappeared'] = 0
            det_center = self.get_center(detections[j])
            self.objects[obj_id]['positions'].append(det_center)
            if len(self.objects[obj_id]['positions']) > max_positions_history:
                self.objects[obj_id]['positions'].pop(0)
            
            matched_obj_indices.add(i)
            matched_det_indices.add(j)
            
            cost_matrix[i, :] = float('inf')
            cost_matrix[:, j] = float('inf')
        
        for j in set(range(len(detections))) - matched_det_indices:
            det = detections[j]
            det_center = self.get_center(det)
            
            width = det[2] - det[0]
            height = det[3] - det[1]
            if width < min_object_size or height < min_object_size:
                continue
            
            too_close = False
            for obj_id in self.objects:
                obj_center = self.get_center(self.objects[obj_id]['bbox'])
                distance = np.linalg.norm(obj_center - det_center)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                self.objects[self.next_id] = {
                    'bbox': det,
                    'disappeared': 0,
                    'positions': [det_center],
                    'predicted_pos': det_center
                }
                self.next_id += 1
        
        return self._prepare_output(current_frame)
    
    def _prepare_output(self, current_frame):
        current_tracks = [
            [obj_id, obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]]
            for obj_id, obj in self.objects.items()
        ]
        
        if current_frame is not None:
            self.metrics.update_metrics(len(self.metrics.metrics['frame']), current_tracks, current_frame)
        return {obj_id: obj['bbox'] for obj_id, obj in self.objects.items()}
    
    @staticmethod
    def get_center(bbox):
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    
    @staticmethod
    def calculate_iou(bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return inter_area / float(bbox1_area + bbox2_area - inter_area)
  
class VideoTracker:
    def __init__(self, model_path, tracker_type):
        self.tracker_type = tracker_type
        
        # Определяем тип модели по tracker_type
        if tracker_type == 'Мой трекер + CounTR':
            self.model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.custom_tracker = ObjectTracker(max_disappeared=max_disappeared)
            self.is_yolo_model = False
        else:
            self.model = YOLO(model_path)
            self.is_yolo_model = True
        
        self.quality_analyzer = TrackingQualityAnalyzer()
        self.processed_frames = []
        
        if tracker_type == 'DeepSORT + YOLOv11s':
            self.deep_sort_tracker = DeepSortTracker(img_size=(640, 480))

    def _parse_tracks(self, results):
        """Унифицированный метод преобразования треков"""
        tracks = []
        
        if self.tracker_type == 'DeepSORT + YOLOv11s':
            if results[0].boxes.data is not None:
                detections = results[0].boxes.data.cpu().numpy()
                track_results = self.deep_sort_tracker.update(detections)
                tracks = [[int(t[0]), t[1], t[2], t[3], t[4]] for t in track_results]
        elif self.tracker_type == 'Мой трекер + CounTR':
            # print('CounTR + Мой трекер')
            if isinstance(results, dict):  # Для custom трекера
                print(1)
                tracks = []
                for obj_id, obj_data in results.items():
                    print(obj_id, obj_data)
                    bbox = obj_data['bbox']  # Получаем bbox из словаря
                    if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
                        # Конвертируем в формат [track_id, x1, y1, x2, y2]
                        tracks.append([obj_id, bbox[0], bbox[1], bbox[2], bbox[3]])
                print(tracks)  # Выводим результат
                return tracks
        else:  # Другие трекеры YOLO
            if hasattr(results[0], 'boxes') and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                tracks = [[ids[i], boxes[i][0], boxes[i][1], boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1]] 
                        for i in range(len(boxes))]
        print(tracks)
        return tracks

    def _draw_detections(self, frame, tracks):
        for track in tracks:
            track_id, x, y, w, h = track
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame

    def non_max_suppression(self, points, scores, radius=nms_radius):
        sorted_indices = np.argsort(scores)[::-1]
        keep = []
        
        while sorted_indices.size > 0:
            i = sorted_indices[0]
            keep.append(i)
            
            dists = np.sqrt(
                (points[i,0] - points[sorted_indices[1:],0])**2 + 
                (points[i,1] - points[sorted_indices[1:],1])**2
            )
            
            to_remove = np.where(dists < radius)[0] + 1
            sorted_indices = np.delete(sorted_indices, [0] + list(to_remove))
        
        return points[keep]

    def process_frame(self, frame, model, old_w, old_h, tracker=None):
        if tracker is None:
            tracker = ObjectTracker(max_disappeared=max_disappeared)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        w, h = pil_img.size
        ratio = min(model_size/w, model_size/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        
        padded_img = Image.new('RGB', (model_size, model_size), (0, 0, 0))
        pad_x = (model_size - new_w) // 2
        pad_y = (model_size - new_h) // 2
        padded_img.paste(pil_img, (pad_x, pad_y))
        
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(padded_img).unsqueeze(0)
        boxes = torch.zeros(1, 0, 4)
        
        with torch.no_grad():
            density_map = model(img_tensor, boxes, shot_num)[0].squeeze(0).cpu().numpy()
        
        y, x = np.where(density_map > density_threshold * density_map.max())
        scores = density_map[y, x]
        points = np.column_stack((x, y))
        
        filtered_points = self.non_max_suppression(points, scores) if len(points) > 0 else []
        
        current_detections = []
        for (x, y) in filtered_points:
            orig_x = int((x - pad_x) * old_w / new_w)
            orig_y = int((y - pad_y) * old_h / new_h)
            
            x1 = max(0, orig_x - bbox_size)
            y1 = max(0, orig_y - bbox_size)
            x2 = min(old_w - 1, orig_x + bbox_size)
            y2 = min(old_h - 1, orig_y + bbox_size)
            current_detections.append((x1, y1, x2, y2))
        
        tracked_objects = tracker.update(current_detections, frame)
        
        for obj_id, bbox in tracked_objects.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(obj_id), (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # metrics = tracker.metrics.get_final_metrics()
        # cv2.putText(frame, f"Total: {len(tracked_objects)}", (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, f"Score: {tracker.metrics.get_tracking_score():.2f}", (10, 60), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        metrics = tracker.metrics.get_final_metrics()
        cv2.putText(frame, f"Total: {len(tracked_objects)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, len(tracked_objects), tracker

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.processed_frames = []
        
        if self.tracker_type == 'DeepSORT + YOLOv11s':
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.deep_sort_tracker = DeepSortTracker(img_size=(frame_width, frame_height))
        
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.tracker_type == 'Мой трекер + CounTR':
                # Обработка кадра с помощью custom трекера
                processed_frame, count, tracker = self.process_frame(
                    frame, self.model, frame.shape[1], frame.shape[0], self.custom_tracker
                )
                current_tracks = self._parse_tracks(tracker.objects)
                frame_with_detections = processed_frame
                
                # Обновляем метрики с правильным форматом данных
                self.quality_analyzer.update_metrics(frame_num, current_tracks, frame)
            else:
                if self.tracker_type == 'DeepSORT + YOLOv11s':
                    results = self.model(frame, conf=0.3, iou=0.5, max_det=400)
                    current_tracks = self._parse_tracks(results)
                else:
                    results = self.model.track(
                        frame,
                        tracker=self.tracker_type.lower().replace(' + yolov11s', '.yaml'),
                        persist=True,
                        conf=0.3,
                        iou=0.5,
                        max_det=400
                    )
                    current_tracks = self._parse_tracks(results)
                
                frame_with_detections = self._draw_detections(frame.copy(), current_tracks)
                self.quality_analyzer.update_metrics(frame_num, current_tracks, frame)
            
            self.processed_frames.append({
                'frame': frame_with_detections,
                'tracks': current_tracks,
                'bubbles_count': len(current_tracks)
            })
            
            progress_bar.progress(cap.get(cv2.CAP_PROP_POS_FRAMES)/total_frames)
            frame_num += 1
        
        processing_time = time.time() - start_time
        cap.release()
        
        metrics = self.quality_analyzer.get_final_metrics()
        metrics['final_score'] = self.quality_analyzer.get_tracking_score()  # Вычисляем один раз
        
        metrics.update({
            'bubbles_per_frame': [f['bubbles_count'] for f in self.processed_frames],
            'max_active_tracks_history': self.quality_analyzer.metrics['active_tracks'],
            'track_lengths': self.quality_analyzer.metrics['track_lengths'],
            'displacement': self.quality_analyzer.metrics['displacement'],
            'coverage': self.quality_analyzer.metrics['coverage'],
            'temporal_consistency': self.quality_analyzer.metrics['temporal_consistency'],
            'optical_flow': self.quality_analyzer.metrics['optical_flow'],
            'processing_time': processing_time
        })
        
        return metrics


def main():
    # Конфигурация страницы
    st.set_page_config(layout="wide")
    st.title("Автоматизация анализа флотации")

    db = TrackingDatabase()

    # Создаем вкладки
    tab1, tab2, tab3 = st.tabs(["Основные метрики", "Покадровый просмотр", "Детальные графики"])

    # Проверка конфигов
    tracker_configs = {
        'ByteTrack + YOLOv11s': 'bytetrack.yaml', 
        'BoT-SORT + YOLOv11s': 'bot-sort.yaml', 
        'DeepSORT + YOLOv11s': None, 
        'Мой трекер + CounTR': None
    }

    # Загрузка видео
    uploaded_file = st.sidebar.file_uploader("Выберите видео", type=["mp4", "avi", "mov"])
    tracker_type = st.sidebar.selectbox("Выберите трекер", list(tracker_configs.keys()))

    # Инициализация состояния сессии
    if 'tracker_results' not in st.session_state:
        st.session_state.tracker_results = None
        st.session_state.processed = False

    if uploaded_file and st.sidebar.button("Запустить оценку"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        with st.spinner("Обработка видео..."):
            if tracker_type == 'Мой трекер + CounTR':
                model_path = 'model/FSC147.pth'
                model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model'], strict=False)
                tracker = VideoTracker(model_path, tracker_type)
                tracker.model = model  # Используем загруженную модель
            else:
                tracker = VideoTracker('model/YOLOv11s.pt', tracker_type)
            metrics = tracker.process_video(video_path)

            print("Debug - metrics keys:", metrics.keys())

            analysis_data = {
                'filename': uploaded_file.name,
                'model_name': tracker_type,
                'avg_displacement': np.mean(metrics['displacement']) if metrics['displacement'] else 0,
                'tracking_coverage': np.mean(metrics['coverage']) if metrics['coverage'] else 0,
                'temporal_consistency': np.mean(metrics['temporal_consistency']) if metrics['temporal_consistency'] else 0,
                'optical_flow': np.mean(metrics['optical_flow']) if metrics['optical_flow'] else 0,
                'avg_track_length': np.mean(list(metrics['track_lengths'].values())) if metrics['track_lengths'] else 0,
                'max_active_tracks': max(metrics['max_active_tracks_history']) if metrics['max_active_tracks_history'] else 0,
                'tracking_score': metrics['final_score'],
                'processing_time': metrics['processing_time']
            }
            db.save_analysis(analysis_data)
            
            # Сохраняем результаты в session_state
            st.session_state.tracker_results = {
                'metrics': metrics,
                'processed_frames': tracker.processed_frames,
                'quality_analyzer': tracker.quality_analyzer
            }
            st.session_state.processed = True
            os.unlink(video_path)

    # Отображение результатов
    if st.session_state.tracker_results and st.session_state.processed:
        metrics = st.session_state.tracker_results['metrics']
        processed_frames = st.session_state.tracker_results['processed_frames']
        quality_analyzer = st.session_state.tracker_results.get('quality_analyzer', None)
        
        with tab1:
            st.success("Обработка завершена!")
            
            cols = st.columns(3)
            with cols[0]:
                # Используем среднее значение из списка displacement
                avg_displacement = np.mean(metrics['displacement']) if metrics['displacement'] else 0
                st.metric("Среднее смещение", f"{avg_displacement:.2f} px")
                
                # Используем среднее значение из списка coverage
                avg_coverage = np.mean(metrics['coverage']) if metrics['coverage'] else 0
                st.metric("Полнота обнаружения", f"{avg_coverage:.2%}")

            with cols[1]:
                # Используем среднее значение из списка optical_flow
                avg_optical_flow = np.mean(metrics['optical_flow']) if metrics['optical_flow'] else 0
                st.metric("Средний оптический поток", f"{avg_optical_flow:.2f} px")
                
                # Используем среднее значение из списка temporal_consistency
                avg_temp_consistency = np.mean(metrics['temporal_consistency']) if metrics['temporal_consistency'] else 0
                st.metric("Темпоральная согласованность", f"{avg_temp_consistency:.2f}")

            with cols[2]:
                # Используем среднее значение длин треков
                avg_track_length = np.mean(list(metrics['track_lengths'].values())) if metrics['track_lengths'] else 0
                st.metric("Средняя длина трека", f"{avg_track_length:.2f} кадров")
                
                # Используем максимум из истории активных треков
                max_active_tracks = max(metrics['max_active_tracks_history']) if metrics['max_active_tracks_history'] else 0
                st.metric("Макс. активных треков", max_active_tracks)

            # Итоговая оценка остается без изменений
            st.metric("Итоговая оценка", f"{metrics['final_score']:.4f}", 
                    help="Оценка от 0 до 1, где 1 - наилучшее качество трекинга")

            # График динамики пузырей
            if 'bubbles_per_frame' in metrics and metrics['bubbles_per_frame']:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Динамика обнаружения пузырей")
                    chart_data = pd.DataFrame({
                        'Кадр': range(len(metrics['bubbles_per_frame'])),
                        'Количество пузырей': metrics['bubbles_per_frame']
                    })
                    st.line_chart(chart_data.set_index('Кадр'))
                
                with col2:
                    st.subheader("Статистика")
                    stats = {
                        'Всего обнаружено': sum(metrics['bubbles_per_frame']),
                        'Максимум в кадре': max(metrics['bubbles_per_frame']),
                        'Среднее значение': round(np.mean(metrics['bubbles_per_frame']), 1),
                        'Пустых кадров': metrics['bubbles_per_frame'].count(0)
                    }
                    for k, v in stats.items():
                        st.metric(k, v)

        with tab2:
            st.header("Покадровый просмотр результатов")
            
            frame_idx = st.slider("Выберите кадр", 0, len(processed_frames)-1, 0)
            frame_data = processed_frames[frame_idx]
            
            st.image(frame_data['frame'], channels="BGR", 
                caption=f"Кадр {frame_idx+1} из {len(processed_frames)}")
            
            st.write(f"Количество пузырей: {frame_data['bubbles_count']}")

        with tab3:
            st.header("Детальные графики метрик")
            
            if quality_analyzer:
                # Используем встроенные графики из анализатора качества
                quality_analyzer.plot_metrics()
            else:
                # Резервный вариант, если анализатора нет
                if 'max_active_tracks_history' in metrics and metrics['max_active_tracks_history']:
                    st.subheader("Активные треки по кадрам")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(metrics['max_active_tracks_history'], 'c-', label='Активные треки')
                    ax.axhline(y=metrics['max_active_tracks'], color='r', linestyle='--', 
                            label=f'Максимум: {metrics["max_active_tracks"]}')
                    ax.set_xlabel("Номер кадра")
                    ax.set_ylabel("Количество треков")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
                
                if 'optical_flow' in metrics and metrics['optical_flow']:
                    st.subheader("Оптический поток")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(metrics['optical_flow'], 'm-', label='Оптический поток')
                    ax.set_xlabel("Номер кадра")
                    ax.set_ylabel("Величина потока (пиксели)")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
                
                if 'track_lengths' in metrics and metrics['track_lengths']:
                    st.subheader("Распределение длин треков")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    
                    lengths = metrics['track_lengths']
                    if isinstance(lengths, dict):
                        lengths = list(lengths.values())
                    elif not isinstance(lengths, (list, np.ndarray)):
                        lengths = []
                    
                    if lengths:
                        ax.hist(lengths, bins=20, color='orange', 
                            edgecolor='black', alpha=0.7)
                        
                        mean_length = np.mean(lengths)
                        ax.axvline(mean_length, color='r', linestyle='--', 
                                label=f'Среднее: {mean_length:.1f} кадров')
                        
                        ax.set_xlabel("Длина трека (кадры)")
                        ax.set_ylabel("Количество треков")
                        ax.grid(True)
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Нет данных о длинах треков для построения гистограммы")

                if 'displacement' in metrics and len(metrics['displacement']) > 0:
                    st.subheader("Среднее смещение объектов между кадрами")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(metrics['displacement'], 'r-', label='Смещение (пиксели)')
                    ax.set_xlabel("Номер кадра")
                    ax.set_ylabel("Смещение")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
                
                if 'coverage' in metrics and len(metrics['coverage']) > 0:
                    st.subheader("Полнота обнаружения")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(metrics['coverage'], 'g-', label='Процент совпадений')
                    ax.set_xlabel("Номер кадра")
                    ax.set_ylabel("Процент")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
                
                if 'temporal_consistency' in metrics and len(metrics['temporal_consistency']) > 0:
                    st.subheader("Темпоральная согласованность (IoU)")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(metrics['temporal_consistency'], 'm-', label='Средний IoU')
                    ax.set_xlabel("Номер кадра")
                    ax.set_ylabel("IoU")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
        

    elif not st.session_state.processed:
        st.info("Загрузите видеофайл и нажмите кнопку 'Запустить оценку'")
    
    st.header("История запусков")
    db.display_recent_analyses()

if __name__ == "__main__":
    main()