import json
import sqlite3
import struct
import threading
from pathlib import Path
from core.VectorEntity import VectorEntity

class VectorDB:
    def __init__(self, db_file: Path, vector_dim=512):
        self.db_file = Path(db_file)# مسیر فایل
        self.vector_dim = vector_dim # ابعاد بردار
        self.lock = threading.Lock() # برای اینکه چند ترد روی بردر کار نکنن ی لاک تعریف میکنمیم
        self.entities = []  # in-memory cache
        self.conn = None
        self._init_sqlite() # ساخت تیبل اس کیو ال
        self.load() # لود کردن

    def add_entity(self, entity: VectorEntity):
        with self.lock:
            self.entities.append(entity)
            labels_json = json.dumps(entity.labels)
            vec_blob = self._pack_vector(entity.vector)
            with self.conn:
                self.conn.execute(
                    "INSERT INTO vectors (id, vector, labels, image_path) VALUES (?, ?, ?, ?)",
                    (entity.id, vec_blob, labels_json, entity.image_path),
                )

    def get_entity(self, entity_id: str):
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None

    def update_entity(self, entity_id: str, new_labels=None, new_image_path=None):
        with self.lock:
            e = self.get_entity(entity_id)
            if e is None:
                return False
            if new_labels is not None:
                e.labels = new_labels
                e.metadata["labels"] = new_labels
            if new_image_path is not None:
                e.image_path = new_image_path
            labels_json = json.dumps(e.labels)
            with self.conn:
                self.conn.execute(
                    "UPDATE vectors SET labels = ?, image_path = ? WHERE id = ?",
                    (labels_json, e.image_path, e.id),
                )
            return True

    def delete_entity(self, entity_id: str):
        with self.lock:
            e = self.get_entity(entity_id)
            if e is None:
                return False
            self.entities.remove(e)
            with self.conn:
                self.conn.execute("DELETE FROM vectors WHERE id = ?", (entity_id,))
            return True

    def load(self):
        self.entities = self._load_from_sqlite()

    def _init_sqlite(self):
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                labels TEXT NOT NULL,
                image_path TEXT NOT NULL
            )
            """
        )
        self.conn.commit()
# تبدیل وکتور بلاب به فلوت
    def _pack_vector(self, vector):
        if len(vector) == 0:
            return b""
        return struct.pack(f"{len(vector)}f", *vector)
# و بالعکس
    def _unpack_vector(self, blob):
        if not blob:
            return []
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))

    def _load_from_sqlite(self):
        entities = []
        cursor = self.conn.execute("SELECT id, vector, labels, image_path FROM vectors")
        for entity_id, vec_blob, labels_json, image_path in cursor.fetchall():
            labels = json.loads(labels_json) if labels_json else []
            vector = self._unpack_vector(vec_blob)
            entity = VectorEntity(vector=vector, labels=labels, image_path=image_path, id=entity_id)
            entities.append(entity)
        return entities


# گرفتن همه وکتور های تیبل
    @property
    def vectors(self):
        return [e.vector for e in self.entities]
# گرفتن همه ایدی ها
    @property
    def ids(self):
        return [e.id for e in self.entities]
# گرفتن همه ادرس ها
    @property
    def image_paths(self):
        return [e.image_path for e in self.entities]
