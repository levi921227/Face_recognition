import mysql.connector
import numpy as np


class FaceRecognitionDatabase:
    def __init__(self, db_config):
        self.conn = mysql.connector.connect(**db_config)
        self.cursor = self.conn.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                embedding BLOB,
                image_path VARCHAR(255),
                group_name VARCHAR(255)
            )
        ''')
        self.conn.commit()

    def save_embedding(self, name, embedding, image_path, group_name):
        embedding_bytes = embedding.tobytes()
        self.cursor.execute('''
            INSERT INTO face_embeddings (name, embedding, image_path, group_name) 
            VALUES (%s, %s, %s, %s)
        ''', (name, embedding_bytes, image_path, group_name))
        self.conn.commit()

    def get_all_embeddings(self):
        self.cursor.execute('SELECT id, name, embedding, image_path, group_name FROM face_embeddings')
        return self.cursor.fetchall()

    def close_connection(self):
        self.cursor.close()
        self.conn.close()
