# object/satellite.py

import asyncio
import torch
import numpy as np
from datetime import datetime, timedelta, timezone
from utils.skyfield_utils import EarthSatellite
from utils.logging_setup import setup_loggers, KST
from typing import Dict, List, Set
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict
from skyfield.api import load, wgs84

# config에서 필요한 변수 가져오기
from config import IOT_FLYOVER_THRESHOLD_DEG, GS_FLYOVER_THRESHOLD_DEG, LOCAL_EPOCHS

# 변경된 모듈 임포트 (CIFAR-10 & ResNet-9)
from ml.data import get_cifar10_loaders
from ml.model import create_resnet9, PyTorchModel
from ml.training import train_model
from ml.aggregation import calculate_mixing_weight, weighted_update

class Satellite_Manager:
    def __init__(self, start_time: datetime, end_time: datetime, sim_logger, perf_logger):
        self.start_time = start_time
        self.end_time = end_time

        self.sim_logger = sim_logger
        self.perf_logger = perf_logger

        self.satellites: Dict[int, EarthSatellite] = {}
        self.satellite_models: Dict[int, PyTorchModel] = {}
        self.satellite_performances: Dict[int, float] = {}
        
        # [Topology 관리]
        self.masters: Set[int] = set()           # Master 위성 ID 집합
        self.workers: Set[int] = set()           # Worker 위성 ID 집합

        self.masters_data_count = defaultdict(int)

        self.check_arr = defaultdict(list)

        # --- [FL 설정] ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_satellites = 50
        self.NUM_CLASSES = 10 

        self.sim_logger.info("CIFAR-10 데이터셋 로드 및 분할 중...")
        
        self.avg_data_count, self.client_subsets, self.val_loader, _ = get_cifar10_loaders(
            num_clients=self.num_satellites, 
            dirichlet_alpha=0.5,
            data_root='./data' 
        )
        self.sim_logger.info(f"데이터셋 로드 완료. 위성당 평균 데이터 수: {self.avg_data_count:.1f}")

        # 글로벌 모델 초기화
        self.global_model_net = create_resnet9(num_classes=self.NUM_CLASSES)
        self.global_model_net.to('cpu') 

        self.global_model_wrapper = PyTorchModel.from_model(self.global_model_net, version=0.0)
        self.best_acc = 0.0

        self.sim_logger.info("위성 관리자 생성 완료.")

    def load_constellation(self):
        tle_path = "constellation.tle"
        satellites = {}
        try:
            with open(tle_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
                i = 0
                while i < len(lines):
                    if not lines[i]: 
                        i += 1
                        continue
                    name, line1, line2 = lines[i:i+3]
                    sat_id = int(name.replace("SAT", "").replace("_", ""))
                    satellites[sat_id] = EarthSatellite(line1, line2, name)
                    i += 3
            self.satellites = satellites
            
            # [Topology 구성] 
            # 각 궤도면의 0번 위성(0, 10, 20, 30, 40)을 Master로 지정
            for sat_id in self.satellites.keys():
                if sat_id % 10 == 0:
                    self.masters.add(sat_id)
                else:
                    self.workers.add(sat_id)
            
            for master_id in self.masters:
                loader_idx = master_id % len(self.client_subsets)
                self.masters_data_count[master_id] = len(self.client_subsets[loader_idx])

            self.sim_logger.info(f"Constellation Topology: {len(self.masters)} Masters, {len(self.workers)} Workers.")
            self.sim_logger.info("Topology Mode: Dynamic (Inter-plane & Intra-plane allowed)")
            
        except Exception as e:
            self.sim_logger.error(f"TLE 파일 로드 실패: {e}")
            raise e

    async def run(self):
        self.sim_logger.info("위성 관리자 운영 시작.")
        self.load_constellation()

        # 초기 모델 배포
        for sat_id in self.satellites.keys():
            self.satellite_models[sat_id] = PyTorchModel.from_model(self.global_model_net, version=0.0)
            self.satellite_performances[sat_id] = 0.0

        await self.propagate_orbit(self.start_time, self.end_time)
        self.sim_logger.info(f"궤도 전파 완료 ({len(self.times)} steps).")

        await self.check_iot_comm()
        await self.check_master_comm_optimized() # [Layer 1] 최적화된 Inter-plane 통신 분석
        await self.check_gs_comm()               # [Layer 2] Global 통신 분석
        self.sim_logger.info("모든 통신 스케줄 계산 완료.")

        await self.manage_fl_process()
        self.sim_logger.info("모든 시뮬레이션 종료.")

    async def propagate_orbit(self, start_time, end_time):
        step = timedelta(seconds=10)
        self.times = []
        curr = start_time
        while curr < end_time:
            self.times.append(curr)
            curr += step

        ts = load.timescale() 
        self.t_vector = ts.from_datetimes(self.times)

    async def check_iot_comm(self):
        self.sim_logger.info("IoT 통신 가능 시간 분석 시작...")
        iot_devices = [
            {"name": "Amazon_Forest", "loc": wgs84.latlon(-3.47, -62.37, elevation_m=100)},
            {"name": "Great_Barrier_Reef", "loc": wgs84.latlon(-18.29, 147.77, elevation_m=0)},
            {"name": "Abisko Tundra", "loc": wgs84.latlon(68.35, 18.79, elevation_m=420)},
        ]    
        for iot in iot_devices:
            for sat_id, satellite in self.satellites.items():
                difference = satellite - iot['loc']
                topocentric = difference.at(self.t_vector)
                alt, _, _ = topocentric.altaz()
                visible_indices = np.where(alt.degrees > IOT_FLYOVER_THRESHOLD_DEG)[0]
                if len(visible_indices) == 0: continue
                
                breaks = np.where(np.diff(visible_indices) > 1)[0] + 1
                windows = np.split(visible_indices, breaks)
                
                for window in windows:
                    start_idx = window[0]
                    end_idx = window[-1]
                    duration = (self.times[end_idx] - self.times[start_idx]).total_seconds()
                    if duration == 0: duration = 10
                    
                    event = {
                        "type": "IOT_TRAIN",
                        "start_time": self.times[start_idx],
                        "end_time": self.times[end_idx],
                        "duration": duration,
                        "target": iot['name']
                    }
                    self.check_arr[sat_id].append(event)

    async def check_master_comm_optimized(self):
        """
        [Layer 1 최적화] 모든 Worker <-> 모든 Master 간의 통신 분석
        Intra-plane + Inter-plane 모두 포함
        """
        self.sim_logger.info("Layer 1 (Intra & Inter-plane) 통신 분석 시작 (Optimized)...")
        
        ISL_THRESHOLD_KM = 5000.0 # 표준 ISL 통신 거리 복구 (5000km)
        
        # 1. 모든 위성의 위치 좌표 미리 계산 (메모리 사용량 증가하지만 속도 대폭 향상)
        # Shape: (Num_Sats, 3, Num_Steps)
        self.sim_logger.info("  - 위성 위치 좌표 사전 계산 중...")
        positions = {}
        for sat_id, sat in self.satellites.items():
            positions[sat_id] = sat.at(self.t_vector).position.km
            
        master_ids = list(self.masters)
        worker_ids = list(self.workers)
        
        # 2. Worker별로 모든 Master와의 거리 계산 (Broadcasting)
        self.sim_logger.info(f"  - {len(worker_ids)} Workers x {len(master_ids)} Masters 거리 계산 중...")
        
        for worker_id in worker_ids:
            w_pos = positions[worker_id] # Shape: (3, Steps)
            
            for master_id in master_ids:
                m_pos = positions[master_id] # Shape: (3, Steps)
                
                # 유클리드 거리 계산 (Vectorized)
                # dists shape: (Steps,)
                dists = np.linalg.norm(w_pos - m_pos, axis=0)
                
                # 통신 가능 구간 추출
                visible_indices = np.where(dists < ISL_THRESHOLD_KM)[0]
                if len(visible_indices) == 0: continue
                
                breaks = np.where(np.diff(visible_indices) > 1)[0] + 1
                windows = np.split(visible_indices, breaks)
                
                for window in windows:
                    start_idx = window[0]
                    end_idx = window[-1]
                    duration = (self.times[end_idx] - self.times[start_idx]).total_seconds()
                    if duration == 0: duration = 10
                    
                    # Worker -> Any Master 접속 이벤트
                    event = {
                        "type": "MASTER_AGGREGATE",
                        "start_time": self.times[start_idx],
                        "end_time": self.times[end_idx],
                        "duration": duration,
                        "target": f"Master_{master_id}", # 접속 대상 Master (동적)
                        "master_id": master_id
                    }
                    self.check_arr[worker_id].append(event)

    async def check_gs_comm(self):
        """[Layer 2] Master <-> GS 간의 통신 분석"""
        self.sim_logger.info("Layer 2 (Master-GS) 통신 분석 시작...")
        gs = {"name": "Ground Station", "loc": wgs84.latlon(37.5665, 126.9780, elevation_m=34)}
        
        target_satellites = self.masters
        
        for sat_id in target_satellites:
            satellite = self.satellites[sat_id]
            difference = satellite - gs['loc']
            topocentric = difference.at(self.t_vector)
            alt, _, _ = topocentric.altaz()
            
            visible_indices = np.where(alt.degrees > GS_FLYOVER_THRESHOLD_DEG)[0]
            if len(visible_indices) == 0: continue
            
            breaks = np.where(np.diff(visible_indices) > 1)[0] + 1
            windows = np.split(visible_indices, breaks)
            
            for window in windows:
                start_idx = window[0]
                end_idx = window[-1]
                duration = (self.times[end_idx] - self.times[start_idx]).total_seconds()
                if duration == 0: duration = 10
                
                event = {
                    "type": "GS_AGGREGATE",
                    "start_time": self.times[start_idx],
                    "end_time": self.times[end_idx],
                    "duration": duration,
                    "target": gs['name']
                }
                self.check_arr[sat_id].append(event)

    def _evaluate_direct(self, model, data_loader, sat_id, version, stage):
        model.to(self.device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct = 0; total = 0; total_loss = 0.0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        
        self.perf_logger.info(
            f"{datetime.now(KST).isoformat()},{stage},{sat_id},{version},N/A,{acc:.4f},{avg_loss:.6f},0.0000"
        )
        return acc, avg_loss

    async def manage_fl_process(self):
        self.sim_logger.info("\n=== 2-Layer 계층적 비동기 연합 학습 (Inter/Intra Plane) 시작 ===")
        
        all_events = []
        for sat_id, events in self.check_arr.items():
            for event in events:
                event['sat_id'] = sat_id
                all_events.append(event)

        all_events.sort(key=lambda x: x['start_time'])
        [self.sim_logger.info(i) for i in all_events]
        
        self.sim_logger.info(f"📅 총 {len(all_events)}개의 이벤트 처리 시작")
        
        temp_model = create_resnet9(num_classes=self.NUM_CLASSES)
        avg_cluster_data_count = self.avg_data_count * (self.num_satellites / len(self.masters))

        for i, event in enumerate(all_events):
            sat_id = event['sat_id']
            current_local_wrapper = self.satellite_models[sat_id]
            
            # -----------------------------------------------------------
            # [Event 1] 로컬 학습 (Master/Worker 공통)
            # -----------------------------------------------------------
            if event['type'] == 'IOT_TRAIN':
                loader_idx = sat_id % len(self.client_subsets)
                dataset = self.client_subsets[loader_idx]
                train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
                
                current_local_wrapper.to_device(temp_model, device='cpu')
                
                train_model(
                    model=temp_model,
                    global_state_dict=self.global_model_wrapper.model_state_dict, 
                    train_loader=train_loader,
                    epochs=LOCAL_EPOCHS,
                    lr=0.005,
                    device=self.device,
                    sim_logger=self.sim_logger
                )
                
                new_ver = round(current_local_wrapper.version + 0.1, 2)
                acc, _ = self._evaluate_direct(temp_model, self.val_loader, sat_id, new_ver, "LOCAL_TRAIN")
                self.satellite_performances[sat_id] = acc
                
                current_local_wrapper = PyTorchModel.from_model(temp_model, version=new_ver)
                self.satellite_models[sat_id] = current_local_wrapper
                
                self.sim_logger.info(f"📡 [IoT] SAT_{sat_id} Trained -> v{new_ver:.2f} (Acc: {acc:.2f}%)")

            # -----------------------------------------------------------
            # [Event 2] Layer 1 Aggregation (Worker -> Any Master)
            # -----------------------------------------------------------
            elif event['type'] == 'MASTER_AGGREGATE':
                master_id = event['master_id'] # 접속한 Master (자기 궤도면 아니어도 됨)
                master_wrapper = self.satellite_models[master_id]
                
                # 1. Master가 Worker 모델을 Aggregation (Cluster Update)
                local_acc = self.satellite_performances[sat_id]
                loader_idx = sat_id % len(self.client_subsets)
                local_data_count = len(self.client_subsets[loader_idx])
                
                # alpha = 0.2
                alpha, _, _, _ = calculate_mixing_weight(
                    local_ver=current_local_wrapper.version,
                    global_ver=self.global_model_wrapper.version,
                    local_acc=local_acc,
                    global_acc=self.best_acc,
                    local_data_count=local_data_count,
                    avg_data_count=self.avg_data_count
                )
                new_master_state = weighted_update(
                    master_wrapper.model_state_dict,
                    current_local_wrapper.model_state_dict,
                    alpha
                )
                self.masters_data_count[master_id] += local_data_count
                
                # Master 버전 업데이트
                new_master_ver = max(master_wrapper.version, current_local_wrapper.version) + 0.01
                
                master_wrapper = PyTorchModel(
                    version=new_master_ver,
                    model_state_dict=new_master_state,
                    trained_by=master_wrapper.trained_by + [sat_id]
                )
                
                m_acc, m_loss = self._evaluate_direct(temp_model, self.val_loader, f"Master-SAT{master_id}", new_master_ver, "MASTER_TEST")

                self.satellite_models[master_id] = master_wrapper
                self.satellite_performances[master_id] = m_acc

                # 2. Worker가 Master 모델을 Sync (Downlink)
                # Worker는 접속한 Master의 모델을 복제해감 (지식 전파)
                current_local_wrapper = PyTorchModel(
                    version=new_master_ver,
                    model_state_dict={k: v.clone() for k, v in new_master_state.items()},
                    trained_by=[]
                )
                self.satellite_models[sat_id] = current_local_wrapper
                
                self.sim_logger.info(f"🔗 [Layer1] SAT_{sat_id} <-> Master_{master_id} Sync (v{new_master_ver:.2f}), Acc: {m_acc:.2f}%")

            # -----------------------------------------------------------
            # [Event 3] Layer 2 Aggregation (Master -> GS)
            # -----------------------------------------------------------
            elif event['type'] == 'GS_AGGREGATE':
                if self.global_model_wrapper.version > current_local_wrapper.version + 5.0:
                     current_local_wrapper = PyTorchModel.from_model(self.global_model_net, version=self.global_model_wrapper.version)
                     self.satellite_models[sat_id] = current_local_wrapper
                     self.sim_logger.info(f"📥 [Global] Master_{sat_id} Forced Sync -> v{self.global_model_wrapper.version}")
                     continue

                master_acc = self.satellite_performances[sat_id]
                master_data_count = self.masters_data_count[sat_id]
                
                # alpha = 0.2
                alpha, _, _, _ = calculate_mixing_weight(
                    local_ver=current_local_wrapper.version,
                    global_ver=self.global_model_wrapper.version,
                    local_acc=master_acc,
                    global_acc=self.best_acc,
                    local_data_count=master_data_count,
                    avg_data_count=avg_cluster_data_count
                )
                new_global_state = weighted_update(
                    self.global_model_wrapper.model_state_dict,
                    current_local_wrapper.model_state_dict,
                    alpha
                )
                
                new_global_ver = int(self.global_model_wrapper.version) + 1.0
                
                temp_model.load_state_dict(new_global_state)
                g_acc, g_loss = self._evaluate_direct(temp_model, self.val_loader, "GS", new_global_ver, "GLOBAL_TEST")
                
                if g_acc > self.best_acc:
                    self.best_acc = g_acc
                    save_dir = Path("./checkpoints")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'model': new_global_state, 'acc': g_acc
                    }, save_dir / f"global_v{int(new_global_ver)}_acc{g_acc:.2f}.pth")
                    self.sim_logger.info(f"💾 New Global Best: {g_acc:.2f}%")

                self.global_model_wrapper = PyTorchModel(
                    version=new_global_ver,
                    model_state_dict=new_global_state,
                    trained_by=self.global_model_wrapper.trained_by + [sat_id]
                )
                self.global_model_net.load_state_dict(new_global_state)
                
                current_local_wrapper = PyTorchModel.from_model(temp_model, version=new_global_ver)
                self.satellite_models[sat_id] = current_local_wrapper
                
                self.sim_logger.info(f"⚡ [Layer2] Master_{sat_id} <-> GS Sync (Global v{new_global_ver:.0f}, Acc: {g_acc:.2f}%)")

        self.sim_logger.info("\n=== 시뮬레이션 종료 ===")
        self.sim_logger.info(f"Final Global Model Accuracy: {self.best_acc:.2f}%")

def main():
    try:
        start_time = datetime.now(timezone.utc)
        sim_logger, perf_logger = setup_loggers()
        sat_manager = Satellite_Manager(start_time, start_time + timedelta(days=30), sim_logger, perf_logger)
        asyncio.run(sat_manager.run())
    except KeyboardInterrupt:
        print("\n시뮬레이션을 종료합니다.")
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()