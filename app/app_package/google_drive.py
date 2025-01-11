from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle
import io


class GoogleDrive:
    def __init__(self):
        self.creds = None
        self.service = self.authenticate()

    def authenticate(self):
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                self.creds = pickle.load(token)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', ['https://www.googleapis.com/auth/drive.readonly']
                )
                self.creds = flow.run_local_server(port=0)

            with open('token.pickle', 'wb') as token:
                pickle.dump(self.creds, token)

        return build('drive', 'v3', credentials=self.creds)

    def list_images(self):
        results = self.service.files().list(
            q="mimeType contains 'image/'",
            pageSize=10,
            fields="files(id, name)"
        ).execute()
        return results.get('files', [])

    def download_image(self, file_id, file_name):
        request = self.service.files().get_media(fileId=file_id)
        file_path = os.path.join(os.getcwd(), file_name)

        with io.FileIO(file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

        return file_path
