# test_vector_db.py
from core.VectorEntity import VectorEntity
from infrastructure.VectorDB import VectorDB
import numpy as np

def test_vector_db():

    db = VectorDB()
    db.load_from_file()


    e1 = VectorEntity([1,2,3], ["cat"], "cat.jpg")
    e2 = VectorEntity([4,5,6], ["dog"], "dog.jpg")
    db.add_entity(e1)
    db.add_entity(e2)

    print("AFTER add_entity:")
    for ent in db.get_all_entities():
        print(ent.id, ent.labels, ent.vector)

    # --- Read ---
    ent = db.get_by_id(e1.id)
    print("READING E1:")
    print(ent.id, ent.labels, ent.vector)

    # --- Update ---
    new_data = VectorEntity(None, ["kitten"], None)
    updated = db.update(e1.id, new_data)
    print("AFTER update:")
    print(updated.id, updated.labels, updated.vector)

    # --- Delete ---
    db.delete(e2.id)
    print("After delete e2:")
    for ent in db.get_all_entities():
        print(ent.id, ent.labels, ent.vector)

if __name__ == "__main__":
    test_vector_db()
