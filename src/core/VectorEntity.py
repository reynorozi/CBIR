import uuid
import json


class VectorEntity:
    def __init__(self, vector: list, labels=None, image_path: str = "", id=None):
        self.id = id or str(uuid.uuid4())
        self.vector = vector  # list of float
        self.labels = labels or []
        self.image_path = image_path


        self.metadata = {"labels": self.labels}

    def serialize(self) -> bytes:
        """
        Serialize entity to bytes for storage
        """
        import struct

        # تبدیل float ها به بایت دقیق (۴ بایت برای هر float)
        vec_bytes = b"".join([struct.pack("f", f) for f in self.vector])

        metadata_bytes = json.dumps(self.metadata).encode("utf-8")
        image_path_bytes = self.image_path.encode("utf-8")
        id_bytes = self.id.encode("utf-8")

        # ذخیره طول هر بخش + داده‌ها
        data = (
            len(id_bytes).to_bytes(2, "little") + id_bytes +
            len(vec_bytes).to_bytes(4, "little") + vec_bytes +
            len(metadata_bytes).to_bytes(4, "little") + metadata_bytes +
            len(image_path_bytes).to_bytes(2, "little") + image_path_bytes
        )
        return data

    @staticmethod
    def deserialize(data: bytes, vector_dim: int):
        import struct

        offset = 0

        # خواندن ID
        id_len = int.from_bytes(data[offset:offset + 2], "little")
        offset += 2
        entity_id = data[offset:offset + id_len].decode("utf-8")
        offset += id_len

        # خواندن Vector
        vec_len = int.from_bytes(data[offset:offset + 4], "little")
        offset += 4

        vector = []
        for i in range(0, vec_len, 4):
            vector.append(struct.unpack("f", data[offset + i:offset + i + 4])[0])
        offset += vec_len

        # خواندن Metadata
        meta_len = int.from_bytes(data[offset:offset + 4], "little")
        offset += 4

        metadata = json.loads(data[offset:offset + meta_len].decode("utf-8"))
        offset += meta_len

        # خواندن Image Path
        img_len = int.from_bytes(data[offset:offset + 2], "little")
        offset += 2

        image_path = data[offset:offset + img_len].decode("utf-8")
        offset += img_len

        # ساخت entity
        entity = VectorEntity(
            vector=vector,
            labels=metadata.get("labels", []),
            image_path=image_path,
            id=entity_id
        )
        entity.metadata = metadata
        return entity

