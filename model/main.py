from face_package.utils import compare_faces
from face_package.train import train_model

if __name__ == "__main__":
    # 設置數據路徑
    data_folder = "D:/Desktop/Pic/data/"
    save_dir = "face_recognition_models"

    try:
        # 訓練模型
        embedding_model, history = train_model(data_folder, save_dir)
        print("train model finished")

        # 測試模型
        test_image1 = "D:/Desktop/Pic/data/negative01.jpg"
        test_image2 = "D:/Desktop/Pic/data/positive01.jpg"

        similarity = compare_faces(test_image1, test_image2, embedding_model)
        if similarity is not None:
            print(f"face similarity: {similarity:.4f}")
            print(f"相似度閾值建議: 0.7 (大於此值可能是同一個人)")

    except Exception as e:
        print(f"Error: {str(e)}")