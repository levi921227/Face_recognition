import numpy as np
from keras.models import load_model
from .preprocess import preprocess_image


class FaceRecognition:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def generate_embedding(self, image_path):
        face = preprocess_image(image_path)
        face_batch = np.expand_dims(face, axis=0)
        embedding = self.model.predict(face_batch)[0]
        return embedding

    def find_match(self, input_embedding, stored_faces, threshold=0.8):
        best_match = None
        best_similarity = 0

        for face_id, name, embedding_bytes, image_path, group in stored_faces:
            stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            similarity = np.dot(input_embedding, stored_embedding)

            if similarity > best_similarity and similarity > threshold:
                best_match = {'id': face_id, 'name': name, 'similarity': similarity,
                              'image_path': image_path, 'group': group}
                best_similarity = similarity

        return best_match
