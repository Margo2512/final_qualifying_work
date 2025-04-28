import time
import cv2
import os
from ultralytics import YOLO
import numpy as np
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
import matplotlib.pyplot as plt
import torch
import streamlit as st
import sqlite3
import tempfile
import pandas as pd

class YOLODetector:
    def __init__(self, model_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)

    def detect(self, frame, max_det=1000, resize_factor=1):
        start_time = time.time()

        if resize_factor != 1.0:
            h, w = frame.shape[:2]
            frame_resized = cv2.resize(frame, (int(w * resize_factor), int(h * resize_factor)))
        else:
            frame_resized = frame

        results = self.model(frame_resized, max_det=max_det)
        inference_time = time.time() - start_time

        return results, inference_time

    @staticmethod
    def get_detections_array(results):
        return results[0].boxes.data.cpu().numpy()


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
        current_dict = {track[0]: track[1:] for track in current_tracks}

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

    def get_tracking_score(self, weights=None, normalize=True, reference_score=22.5190, max_score=1.0):
        final_metrics = self.get_final_metrics()

        if not final_metrics:
            print("Предупреждение: нет метрик для расчета score!")
            return 0.0

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

            normalized_score = raw_score / reference_score
            
            return max(0.0, min(max_score, normalized_score))
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

        # График смещения и покрытия
        plt.subplot(3, 2, 3)
        plt.plot(self.metrics['frame'], self.metrics['displacement'])
        plt.plot(self.metrics['frame'], self.metrics['coverage'])
        plt.title('Качество трекинга')
        plt.grid(True)

        # График IoU
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
            c.execute('''CREATE TABLE IF NOT EXISTS analyses
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          название_файла TEXT,
                          название_модели TEXT,
                          среднее_смещение REAL,
                          полнота_трекинга REAL,
                          временная_согласованность REAL,
                          оптический_поток REAL,
                          средняя_длина_трека REAL,
                          макс_активных_треков INTEGER,
                          оценка_трекинга REAL,
                          время_обработки REAL,
                          временная_метка DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit()
    
    def save_analysis(self, data):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO analyses 
                         (название_файла, название_модели, среднее_смещение, полнота_трекинга, 
                          временная_согласованность, оптический_поток, 
                          средняя_длина_трека, макс_активных_треков, оценка_трекинга, время_обработки)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                      (data['название_файла'], data['название_модели'], data['среднее_смещение'], 
                       data['полнота_трекинга'], data['временная_согласованность'], 
                       data['оптический_поток'], data['средняя_длина_трека'], 
                       data['макс_активных_треков'], data['оценка_трекинга'], data['время_обработки']))
            conn.commit()
    
    def get_recent_analyses(self, limit=10):
        with sqlite3.connect(self.db_name) as conn:
            df = pd.read_sql(f"SELECT * FROM analyses ORDER BY временная_метка DESC LIMIT {limit}", conn)
            return df
    
    def display_recent_analyses(self):
        df = self.get_recent_analyses()
        st.dataframe(
            df.drop(columns=['id']).style.format({
                'среднее_смещение': "{:.2f}",
                'полнота_трекинга': "{:.2%}",
                'временная_согласованность': "{:.2%}",
                'оценка_трекинга': "{:.2f}"
            }),
            use_container_width=True
        )

class VideoProcessor:
    def __init__(self, model_path, video_path, output_path, detections_folder, yolo_conf=0.5, tracker=None):
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.detections_folder = detections_folder
        self.yolo_conf = yolo_conf
        self.save_images = False
        self.detector = YOLODetector(model_path)
        self.tracker = tracker if tracker else DeepSortTracker((640, 480))
        self.quality_analyzer = TrackingQualityAnalyzer()
        self.video_writer = None
        self.cap = None

        # Статистика
        self.frame_number = 0
        self.total_time = 0
        self.total_processing_time = 0  # Общее время детекции + трекинга
        self.frame_processing_time = 0
        self.frame_count = 0

        # Динамика объектов
        self.new_ids_list = []
        self.gone_ids_list = []
        self.frame_indices = []
        self.changes_per_frame = []
        self.active_ids_prev = set()

    def initialize_components(self):
        self.detector = YOLODetector(self.model_path)
        self.quality_analyzer = TrackingQualityAnalyzer()

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("Ошибка при открытии видео")

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.tracker = DeepSortTracker((frame_width, frame_height))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        os.makedirs(self.detections_folder, exist_ok=True)

    def process_frame(self, frame):
        results, inference_time = self.detector.detect(frame)
        annotations = YOLODetector.get_detections_array(results)

        processed_frame = frame.copy()
        if annotations.size > 0:
            min_confidence = 0.5
            annotations = annotations[annotations[:, 4] >= min_confidence]
            if annotations.size > 0:
                start_tracking_time = time.time()
                track_results = self.tracker.update(annotations)
                tracking_time = time.time() - start_tracking_time

                detections_path = os.path.join(self.detections_folder, f"frame_{self.frame_number:05d}.txt")
                with open(detections_path, "w") as f:
                    active_ids_curr = set()

                    for track_box in track_results:
                        track_id, x1, y1, width, height = track_box
                        active_ids_curr.add(track_id)

                        f.write(f"{int(track_id)},{x1},{y1},{width},{height}\n")

                        cv2.rectangle(processed_frame, (int(x1), int(y1)),
                                    (int(x1 + width), int(y1 + height)), (0, 255, 0), 2)
                        cv2.putText(processed_frame, f'ID: {track_id}', (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    self._analyze_object_dynamics(active_ids_curr)
                    self.quality_analyzer.update_metrics(self.frame_number, track_results, frame)
                    self.frame_processing_time = tracking_time + inference_time
                    self.total_processing_time += self.frame_processing_time

        return processed_frame, inference_time,  self.frame_processing_time

    def _analyze_object_dynamics(self, active_ids_curr):
        new_ids = active_ids_curr - self.active_ids_prev
        gone_ids = self.active_ids_prev - active_ids_curr

        self.new_ids_list.append(len(new_ids))
        self.gone_ids_list.append(len(gone_ids))
        self.frame_indices.append(self.frame_number)
        self.changes_per_frame.append(len(new_ids) + len(gone_ids))
        self.active_ids_prev = active_ids_curr.copy()

    def run(self):
        self.initialize_components()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame, inference_time, frame_processing_time = self.process_frame(frame)
            self.video_writer.write(processed_frame)
            self.total_time += inference_time
            self.frame_count += 1
            print(f"Кадр {self.frame_number}: "
                  f"YOLO = {inference_time:.4f} сек | "
                  f"Детекция+Трекинг = {frame_processing_time:.4f} сек")

            self.frame_number += 1

        self._finalize()

    def _finalize(self):
        try:
            if self.video_writer:
                self.video_writer.release()
            if self.cap:
                self.cap.release()

            self._visualize_results()

        except Exception as e:
            print(f"Ошибка при освобождении ресурсов: {e}")

    def _analyze_object_dynamics_stats(self):
        average_change = np.mean(self.changes_per_frame)
        variance_changes = np.var(self.changes_per_frame)
        total_bubbles = np.array(self.new_ids_list) + np.array(self.gone_ids_list)
        dynamism = np.sum(self.changes_per_frame) / np.sum(total_bubbles) if np.sum(total_bubbles) > 0 else 0
        
        return {
            'average_change': average_change,
            'variance_changes': variance_changes,
            'dynamism': dynamism
        }

    def _visualize_results(self, display_in_streamlit=False):
        graphs_dir = os.path.join(os.path.dirname(self.output_path), 'графики')
        os.makedirs(graphs_dir, exist_ok=True)
        
        self.quality_analyzer.generate_metrics_plots(
            save_path=os.path.join(graphs_dir, 'tracking_quality_metrics.png')
        )
        
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(self.frame_indices, self.new_ids_list, color='green')
        plt.plot(self.frame_indices, self.gone_ids_list, color='red')
        plt.title('Динамика появления и исчезновения объектов')
        plt.grid(True)
        plt.savefig(os.path.join(graphs_dir, 'objects_dynamics.png'))
        
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(self.frame_indices[:len(self.changes_per_frame)], self.changes_per_frame, color='blue')
        plt.title('Изменения объектов по кадрам')
        plt.grid(True)
        plt.savefig(os.path.join(graphs_dir, 'objects_changes.png'))
        
        if display_in_streamlit:
            st.subheader("Графики анализа трекинга")
            st.pyplot(self.quality_analyzer.generate_metrics_plots())
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig1)
            with col2:
                st.pyplot(fig2)
        
        plt.close('all')
    
class StreamlitVideoProcessor:
    def __init__(self, model_path, video_path, output_dir, detections_dir):
        self.model_path = model_path
        self.video_path = video_path
        self.output_dir = output_dir
        self.detections_dir = detections_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(detections_dir, exist_ok=True)
        
        output_video_path = os.path.join(output_dir, "output_video.mp4")
        self.processor = VideoProcessor(
            model_path=model_path,
            video_path=video_path,
            output_path=output_video_path,
            detections_folder=detections_dir
        )
        
    def run_analysis(self):
        try:
            self.processor.initialize_components()
            
            cap = cv2.VideoCapture(self.processor.video_path)
            fps_input_video = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                ret, frame = self.processor.cap.read()
                if not ret:
                    break
                    
                processed_frame, inference_time, frame_time = self.processor.process_frame(frame)
                self.processor.video_writer.write(processed_frame)
                self.processor.frame_number += 1
                self.processor.total_processing_time += frame_time
                
                progress = self.processor.frame_number / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Обработка кадра {self.processor.frame_number}/{total_frames}")
            
            metrics = self.processor.quality_analyzer.get_final_metrics()
            tracking_score = self.processor.quality_analyzer.get_tracking_score()
            obj_dynamics = self.processor._analyze_object_dynamics_stats()
            
            times = {
                'total_time': self.processor.total_processing_time,
                'avg_time': self.processor.total_processing_time / max(1, self.processor.frame_number),
                'fps': self.processor.frame_number / max(1, self.processor.total_processing_time)
            }
            
            self.processor._finalize()
            
            return {
                'metrics': metrics,
                'tracking_score': tracking_score,
                'obj_dynamics': obj_dynamics,
                'times': times,
                'total_frames': total_frames,
                'output_video': self.processor.output_path,
                'detections_dir': self.processor.detections_folder,
                'analyzer': self.processor.quality_analyzer,
                'processor': self.processor,
                'fps_input_video': fps_input_video
            }
            
        except Exception as e:
            st.error(f"Ошибка при обработке видео: {str(e)}")
            if hasattr(self, 'processor'):
                self.processor._finalize()
            return None

def main():
    st.title("Автоматизация анализа флотации")
    st.markdown("---")
    
    db = TrackingDatabase()

    st.subheader("Загрузите видео для анализа")
    uploaded_file = st.file_uploader("Выберите видеофайл", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        st.session_state.video_path = tfile.name
        st.session_state.fps = cap.get(cv2.CAP_PROP_FPS)
        st.session_state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.duration = st.session_state.total_frames / st.session_state.fps
        cap.release()

        st.subheader("Настройки обработки")
        model_option = st.selectbox(
            "Модель детекции + трекинг",
            ["YOLOv11s_DeepSORT", "CounTR_OpticalFlow", "PseCO_OpticalFlow"],
            index=0
        )

        with st.expander("Дополнительные параметры"):
            conf_threshold = st.slider("Порог уверенности", 0.1, 0.9, 0.5, 0.05)

        video_name = os.path.splitext(uploaded_file.name)[0]
        output_dir = os.path.join("results", video_name)
        output_video_path = os.path.join(output_dir, "output_video.mp4")

        if st.button("Запустить анализ", type="primary"):
            with st.spinner("Обработка видео..."):
                processor = StreamlitVideoProcessor(
                    model_path=f"models/{model_option}.pt",
                    video_path=st.session_state.video_path,
                    output_dir=f"results/{os.path.splitext(uploaded_file.name)[0]}",
                    detections_dir=f"results/{os.path.splitext(uploaded_file.name)[0]}/detections"
                )

                results = processor.run_analysis()

                if results:
                    analysis_data = {
                        'название_файла': uploaded_file.name,
                        'название_модели': model_option,
                        'среднее_смещение': results['metrics']['avg_displacement'],
                        'полнота_трекинга': results['metrics']['avg_coverage'],
                        'временная_согласованность': results['metrics']['avg_temporal_consistency'],
                        'оптический_поток': results['metrics']['avg_optical_flow'],
                        'средняя_длина_трека': results['metrics']['track_length_mean'],
                        'макс_активных_треков': results['metrics']['max_active_tracks'],
                        'оценка_трекинга': results['tracking_score'],
                        'время_обработки': results['times']['total_time']
                    }
                    db.save_analysis(analysis_data)

                    st.session_state.results = results
                    st.session_state.output_video_path = output_video_path
                    st.success("Анализ успешно завершен!")

        tab1, tab2, tab3 = st.tabs(["Метрики", "Покадровый просмотр", "Графики"])
       
        with tab1:
            if 'results' in st.session_state:
                fps = st.session_state.results.get('fps_input_video', 0)
                total_frames = st.session_state.results.get('total_frames', 0)
                video_duration = total_frames / fps if fps > 0 else 0

                cols = st.columns(3)
                cols[0].metric("Итоговая оценка", 
                            f"{st.session_state.results['tracking_score']:.2f}",
                            help="Общая оценка качества трекинга")
                cols[1].metric("Кадров обработано", 
                            f"{st.session_state.results['total_frames']}",
                            help="Общее количество кадров")
                cols[2].metric("Длительность видео", 
                            f"{video_duration:.2f} сек",
                            help=f"Исходная длительность ({st.session_state.results['total_frames']} кадров)")

                with st.expander("Динамика объектов"):
                    cols = st.columns(3)
                    cols[0].metric("Среднее изменение объектов", 
                                f"{st.session_state.results['obj_dynamics']['average_change']:.2f}",
                                help="Среднее изменение объектов между кадрами")
                    cols[1].metric("Дисперсия изменений", 
                                f"{st.session_state.results['obj_dynamics']['variance_changes']:.2f}",
                                help="Разброс количества изменений")
                    cols[2].metric("Динамичность", 
                                f"{st.session_state.results['obj_dynamics']['dynamism']:.2f}",
                                help="0=статика, 1=максимальная динамика")
                
                with st.expander("Детальные показатели"):
                    cols = st.columns(3)
                    cols[0].metric("Смещение (px)", 
                                f"{st.session_state.results['metrics']['avg_displacement']:.2f}",
                                help="Среднее смещение объектов")
                    cols[1].metric("Средний оптический поток", 
                                f"{st.session_state.results['metrics']['avg_optical_flow']:.2f}",
                                help="Интенсивность движения")
                    cols[2].metric("Средняя длина трека", 
                                f"{st.session_state.results['metrics']['track_length_mean']:.2f}",
                                help="Средняя продолжительность треков")

                    cols = st.columns(3)
                    cols[0].metric("Максимум активных треков", 
                                f"{st.session_state.results['metrics']['max_active_tracks']}",
                                help="Максимум одновременно отслеживаемых объектов")
                    cols[1].metric("Полнота обнаружений", 
                                f"{st.session_state.results['metrics']['avg_coverage']:.2%}",
                                help="Процент успешно отслеженных объектов")
                    cols[2].metric("Согласованность (IoU)", 
                                f"{st.session_state.results['metrics']['avg_temporal_consistency']:.2%}",
                                help="Стабильность треков между кадрами")

                with st.expander("Производительность"):
                    cols = st.columns(3)
                    cols[0].metric("Общее время", 
                                f"{st.session_state.results['times']['total_time']:.2f} сек",
                                help="Полное время обработки")
                    cols[1].metric("Время на кадр", 
                                f"{st.session_state.results['times']['avg_time']:.4f} сек",
                                help="Среднее время обработки (детекция+трекинг)")
                    cols[2].metric("FPS", 
                                f"{st.session_state.results['times']['fps']:.2f}",
                                help="FPS обработки (детекция+трекинг) - Кадров в секунду")

                st.markdown("---")
                st.caption("Для интерпретации метрик наведите курсор на показатели")

        with tab2:
            if 'results' in st.session_state:
                st.subheader("Покадровый просмотр")

                if 'current_frame' not in st.session_state:
                    st.session_state.current_frame = 0

                st.session_state.current_frame = st.slider(
                    "Выберите кадр",
                    min_value=0,
                    max_value=st.session_state.total_frames - 1,
                    value=st.session_state.current_frame,
                    step=1
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("⏮ Назад", use_container_width=True):
                        st.session_state.current_frame = max(0, st.session_state.current_frame - 1)
                with col2:
                    st.markdown(
                        f"<div style='text-align: center; padding: 0.5rem;'>"
                        f"Кадр: {st.session_state.current_frame + 1} / {st.session_state.total_frames}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with col3:
                    if st.button("⏭ Вперед", use_container_width=True):
                        st.session_state.current_frame = min(st.session_state.total_frames - 1, st.session_state.current_frame + 1)

                if os.path.exists(st.session_state.output_video_path):
                    cap = cv2.VideoCapture(st.session_state.output_video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame, channels="RGB", use_container_width=True)
                    else:
                        st.error("Ошибка чтения кадра!")

        with tab3:
            if 'results' in st.session_state:
                if 'analyzer' in st.session_state.results:
                    st.session_state.results['analyzer'].plot_metrics()
                else:
                    st.warning("Анализатор метрик не найден в результатах")
                
                if 'processor' in st.session_state.results:
                    processor = st.session_state.results['processor']
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        graphs_dir = os.path.join(temp_dir, "graphs")
                        os.makedirs(graphs_dir, exist_ok=True)
                        
                        fig1 = plt.figure(figsize=(12, 6))
                        plt.plot(processor.frame_indices, processor.new_ids_list, color='green')
                        plt.plot(processor.frame_indices, processor.gone_ids_list, color='red')
                        plt.title('Динамика появления и исчезновения объектов')
                        plt.grid(True)
                        st.pyplot(fig1)
                        plt.close(fig1)
                        
                        fig2 = plt.figure(figsize=(12, 6))
                        plt.plot(processor.frame_indices[:len(processor.changes_per_frame)], 
                                processor.changes_per_frame, color='blue')
                        plt.title('Изменения объектов по кадрам')
                        plt.grid(True)
                        st.pyplot(fig2)
                        plt.close(fig2)
                else:
                    st.warning("Данные процессора не найдены в результатах")

    st.subheader("История запусков")
    db.display_recent_analyses()

if __name__ == "__main__":
    main()