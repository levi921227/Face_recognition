from app_package.database import FaceRecognitionDatabase
from app_package.face_recognition import FaceRecognition
from app_package.google_drive import GoogleDrive
from app_package.ui import FaceRecognitionApp
from customtkinter import CTk


def main():
    model_path = r"C:\Users\User\PycharmProjects\Face\Project\Final Project\face_recognition_models\refined_embedding_model.h5"
    db_config = {
        'host': '35.229.213.26',
        'user': 'root',
        'password': '1234',
        'database': 'face_recognition_db'
    }

    # 初始化模組
    face_db = FaceRecognitionDatabase(db_config)
    face_recognition = FaceRecognition(model_path)
    drive = GoogleDrive()
    # embedding = face_recognition.generate_embedding("D:/Desktop/Pic/data/anchor07.jpg")
    # face_db.save_embedding("Chaewon", embedding, "D:/Desktop/Pic/data/anchor07.jpg", "LESSERAFIM")

    try:
        # 啟動 GUI
        root = CTk()
        app = FaceRecognitionApp(root, face_recognition, face_db, drive)
        root.mainloop()
    finally:
        face_db.close_connection()


if __name__ == "__main__":
    main()
