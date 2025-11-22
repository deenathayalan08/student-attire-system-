import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
	import mediapipe as mp
	_MP_AVAILABLE = True
except Exception:
	_MP_AVAILABLE = False

from .config import AppConfig
from .db import get_conn, update_student_face_embedding, get_student_face_embedding


def _get_face_mesh():
	if not _MP_AVAILABLE:
		return None
	return mp.solutions.face_mesh.FaceMesh(
		static_image_mode=True,
		max_num_faces=1,
		refine_landmarks=True,
		min_detection_confidence=0.5,
	)


def _image_to_bgr(image):
	if isinstance(image, np.ndarray):
		return image
	return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@dataclass
class FaceRecognitionResult:
	success: bool
	distance: float
	message: str


class FaceRecognitionSystem:
	def __init__(self, cfg: Optional[AppConfig] = None):
		self.cfg = cfg or AppConfig()
		self.face_mesh = _get_face_mesh()
		self.face_data_file = self.cfg.data_dir / "face_embeddings.json"
		self.face_data_file.parent.mkdir(parents=True, exist_ok=True)

	def _extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
		if not _MP_AVAILABLE or self.face_mesh is None:
			return None

		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		result = self.face_mesh.process(rgb)

		if not result.multi_face_landmarks:
			return None

		landmarks = result.multi_face_landmarks[0].landmark
		embedding = []
		for landmark in landmarks:
			embedding.extend([landmark.x, landmark.y, landmark.z])
		vector = np.array(embedding, dtype=np.float32)
		vector -= np.mean(vector)
		norm = np.linalg.norm(vector)
		if norm == 0:
			return None
		vector /= norm
		return vector

	def register_face(self, student_id: str, image: np.ndarray) -> FaceRecognitionResult:
		bgr = _image_to_bgr(image)
		embedding = self._extract_face_embedding(bgr)
		if embedding is None:
			# Check if MediaPipe is available
			if not _MP_AVAILABLE:
				return FaceRecognitionResult(False, 1.0, "MediaPipe library is not available. Please install it: pip install mediapipe")
			if self.face_mesh is None:
				return FaceRecognitionResult(False, 1.0, "Face detection system is not initialized.")
			return FaceRecognitionResult(False, 1.0, "Unable to detect face in the provided image. Please ensure the face is clearly visible, well-lit, and facing the camera.")

		embedding_list = embedding.astype(float).tolist()
		
		# Store in database
		try:
			update_student_face_embedding(student_id, embedding_list, cfg=self.cfg)
		except Exception as e:
			return FaceRecognitionResult(False, 1.0, f"Failed to save face data to database: {str(e)}")
		
		# Store backup in JSON file
		try:
			self._store_face_embedding_file(student_id, embedding_list)
		except Exception as e:
			print(f"Warning: Failed to save backup file: {e}")
		
		# Verify the data was stored in database
		stored_embedding = get_student_face_embedding(student_id, cfg=self.cfg)
		if stored_embedding is None:
			return FaceRecognitionResult(False, 1.0, "Face embedding was not saved to database. Please try again.")
		
		return FaceRecognitionResult(True, 0.0, "Face registered successfully.")

	def verify_face(self, student_id: str, image: np.ndarray, threshold: float = 0.12) -> FaceRecognitionResult:
		stored_embedding = get_student_face_embedding(student_id, cfg=self.cfg)
		if stored_embedding is None:
			return FaceRecognitionResult(False, 1.0, "No face data registered for this student.")

		bgr = _image_to_bgr(image)
		current_embedding = self._extract_face_embedding(bgr)
		if current_embedding is None:
			return FaceRecognitionResult(False, 1.0, "Unable to detect face in the uploaded image.")

		stored_vector = np.array(stored_embedding, dtype=np.float32)
		current_vector = current_embedding.astype(np.float32)
		if len(stored_vector) != len(current_vector):
			return FaceRecognitionResult(False, 1.0, "Stored face data is incompatible.")

		distance = float(np.linalg.norm(stored_vector - current_vector))
		if distance <= threshold:
			return FaceRecognitionResult(True, distance, "Face matched successfully.")
		return FaceRecognitionResult(False, distance, "Face does not match registered student.")

	def has_face_data(self, student_id: str) -> bool:
		return get_student_face_embedding(student_id, cfg=self.cfg) is not None

	def _store_face_embedding_file(self, student_id: str, embedding: list) -> None:
		try:
			if self.face_data_file.exists():
				with open(self.face_data_file, "r") as f:
					data = json.load(f)
			else:
				data = {}
			data[student_id] = {
				"embedding": embedding,
				"stored_at": self._now_iso(),
			}
			with open(self.face_data_file, "w") as f:
				json.dump(data, f, indent=2)
		except Exception as exc:
			print(f"Warning: Unable to store face embedding file: {exc}")

	@staticmethod
	def _now_iso() -> str:
		from datetime import datetime
		return datetime.now().isoformat()


_FACE_SYSTEM = None


def get_face_system(cfg: Optional[AppConfig] = None) -> FaceRecognitionSystem:
	global _FACE_SYSTEM
	if _FACE_SYSTEM is None:
		_FACE_SYSTEM = FaceRecognitionSystem(cfg)
	return _FACE_SYSTEM

