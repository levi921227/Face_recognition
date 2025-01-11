from tkinter import filedialog, messagebox, simpledialog
from customtkinter import *
from tkinter.ttk import Progressbar
from tkinter import Toplevel, Listbox, Button
from customtkinter import CTkToplevel, CTkButton, CTkLabel, CTkScrollbar


class FaceRecognitionApp:
    def __init__(self, root, face_recognition, face_db, drive):
        self.root = root
        self.face_recognition = face_recognition
        self.face_db = face_db
        self.drive = drive

        self.root.title("人臉辨識系統")
        self.root.geometry('600x400')
        set_appearance_mode("dark")
        self.root.configure(bg='#1E1E1E')

        self.setup_ui()

    def setup_ui(self):
        self.title_label = CTkLabel(
            master=self.root,
            text="Face Recognition System",
            font=("Helvetica", 24, "bold"),
            text_color="#FFCC70"
        )
        self.title_label.pack(pady=10)

        self.google_button = CTkButton(
            master=self.root,
            text="Upload from Google Drive",
            command=self.upload_from_google_drive,
            corner_radius=20,
            fg_color="#FFCC70",
            hover_color="#FFA940",
            text_color="#1E1E1E",
            font=("Arial", 12, "bold"),
            width=200
        )
        self.google_button.pack(pady=15)

        self.upload_button = CTkButton(
            master=self.root,
            text="Upload from your computer",
            command=self.upload_image,
            corner_radius=20,
            fg_color="#FFCC70",
            hover_color="#FFA940",
            text_color="#1E1E1E",
            font=("Arial", 12, "bold"),
            width=200
        )
        self.upload_button.pack(pady=15)

        self.result_label = CTkLabel(
            master=self.root,
            text="Please upload a picture from Google Drive or your computer",
            corner_radius=15,
            font=("Arial", 14),
            fg_color="#2C2C2C",
            text_color="#FFFFFF",
            width=400,
            height=80
        )
        self.result_label.pack(pady=20)

    def upload_from_google_drive(self):
        images = self.drive.list_images()
        if not images:
            messagebox.showinfo("Info", "No images found in Google Drive.")
            return

        file_map = {item['name']: item['id'] for item in images}
        self.show_file_selector(file_map)

    def upload_image(self):
        # 選擇圖片
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[("圖片檔案", "*.jpg;*.jpeg;*.png")]
        )

        if file_path:  # 如果有選擇檔案
            try:
                self.result_label.configure(text="Processing image...", text_color="#FFFFFF")
                self.result_label.update_idletasks()

                # 將選取的檔案路徑傳給 process_image
                self.process_image(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"處理圖片時發生錯誤：{e}")

    def show_file_selector(self, file_map):
        # 創建新窗口
        selector_window = Toplevel(self.root)
        selector_window.title("Select File")
        selector_window.geometry("400x300")

        # 建立 Listbox 顯示檔案清單
        listbox = Listbox(selector_window, width=50, height=15)
        listbox.pack(pady=10)

        # 填充檔案名稱
        for file_name in file_map.keys():
            listbox.insert('end', file_name)

        # 按鈕行為
        def select_file():
            selected_file = listbox.get(listbox.curselection())
            file_id = file_map[selected_file]
            file_path = self.drive.download_image(file_id, selected_file)
            self.process_image(file_path)
            selector_window.destroy()

        # 確認按鈕
        confirm_button = Button(selector_window, text="Confirm", command=select_file)
        confirm_button.pack(pady=5)

    def process_image(self, file_path):
        input_embedding = self.face_recognition.generate_embedding(file_path)
        stored_faces = self.face_db.get_all_embeddings()

        match = self.face_recognition.find_match(input_embedding, stored_faces)

        if match:
            self.result_label.configure(
                text=f"Matched Face: {match['name']}\nSimilarity: {match['similarity']:.4f}\nGroup:{match['group']}",
                text_color="#A7E22E"
            )
        else:
            self.result_label.configure(
                text="No matched face exists",
                text_color="#FF5555"
            )
