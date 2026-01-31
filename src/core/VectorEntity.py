    import uuid
    import json

    class VectorEntity:
        def __init__(self, vector: list, labels=None, image_path: str = "", id=None):
            self.id = id or str(uuid.uuid4())
            self.vector = vector  # list of float
            self.labels = labels or []
            self.image_path = image_path
            self.metadata = {"labels": self.labels}# متا دیتا برای اینکه بعدا اطلاعات عکس رو بتونیم گسترش بدیم

            # labels گفته اگه توی متا دیتا باشه به همون لیبل اشاره کنه

        def serialize(self) -> bytes:
            """
            Serialize entity to bytes for storage
            """

            # از struct  استفاده میکنیم به این دلیل که میخوایم foat دقیقا تبدیل بشه به چهار بایت
            import struct
            vec_bytes = b"".join([struct.pack("f", f) for f in self.vector])
            metadata_bytes = json.dumps(self.metadata).encode("utf-8")
            image_path_bytes = self.image_path.encode("utf-8")
            id_bytes = self.id.encode("utf-8")

    # اینجا ابتدا سایز رو ذخیره میکنیم بعد میایم خودش رو ذخیره میکنیم
            # این little که نوشته شده برای اینه که ترتبیب بایت ها هست که سایزش اول بیاد یا اخر خوده سایز بایت
            # اگه گفت چرا طولو نگه میدارید میگیم برای اینکه بتونیم بعدا برای دی سریالاز کردنش بدونیم از کجا تا کجا مثلا ایدیه از کجا خوده وکتوره و غیر...
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

            # این برای اینه که که بتونه مرحله به مرحله بخونه و جدا کنه و ی اینتیتی بسازه
            id_len = int.from_bytes(data[offset:offset+2], "little")
            offset += 2
            id_bytes = data[offset:offset+id_len]
            offset += id_len
            entity_id = id_bytes.decode("utf-8")

            vec_len = int.from_bytes(data[offset:offset+4], "little")
            offset += 4
            vector = []
            for i in range(0, vec_len, 4):
                vector.append(struct.unpack("f", data[offset + i:offset + i + 4])[0])
            offset += vec_len

            meta_len = int.from_bytes(data[offset:offset+4], "little")
            offset += 4
            metadata_bytes = data[offset:offset+meta_len]
            offset += meta_len
            metadata = json.loads(metadata_bytes.decode("utf-8"))

            img_len = int.from_bytes(data[offset:offset+2], "little")
            offset += 2
            image_path_bytes = data[offset:offset+img_len]
            offset += img_len
            image_path = image_path_bytes.decode("utf-8")

            entity = VectorEntity(vector=vector, labels=metadata.get("labels", []),
                                  image_path=image_path, id=entity_id)
            entity.metadata = metadata
            return entity



