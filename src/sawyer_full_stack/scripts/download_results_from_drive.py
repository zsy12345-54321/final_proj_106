import os
import io

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Define constants for paths and folder names
RESULTS_DRIVE_FOLDER_NAME = 'Image_Processing_Results'  # <--- Modify this to your results folder name in Drive
LOCAL_DOWNLOAD_DIR = 'Processed_Results'  # <--- Modify this to your local download directory
CREDENTIALS_FILE = 'final_proj.json'  # Ensure this file is in the same directory as the script
TOKEN_DOWNLOAD_FILE = 'token_download.json'  # Token file for download authentication

# If modifying these SCOPES, delete the file token_download.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate():
    """Authenticate the user and return the Drive service."""
    creds = None
    # The file token_download.json stores the user's access and refresh tokens for downloading.
    if os.path.exists(TOKEN_DOWNLOAD_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_DOWNLOAD_FILE, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"Missing '{CREDENTIALS_FILE}'. Please follow the setup instructions.")
                exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_DOWNLOAD_FILE, 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    return service

def get_drive_folder_id(service, folder_name):
    """Retrieve the folder ID from Drive by folder name."""
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if items:
        folder_id = items[0]['id']
        print(f"Found folder '{folder_name}' with ID: {folder_id}")
        return folder_id
    else:
        print(f"Folder '{folder_name}' not found in Google Drive.")
        exit(1)

def list_files_in_folder(service, folder_id):
    """List all files in the specified Drive folder."""
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])
    return items

def download_file(service, file_id, destination_path):
    """Download a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    try:
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download {int(status.progress() * 100)}%.")
        print(f"Downloaded file to {destination_path}")
    except Exception as e:
        print(f"Failed to download file ID {file_id}. Error: {e}")

def main():
    # Authenticate and get the Drive service
    service = authenticate()

    # Get the folder ID of the results folder in Drive
    folder_id = get_drive_folder_id(service, RESULTS_DRIVE_FOLDER_NAME)

    # List all files in the results folder
    files = list_files_in_folder(service, folder_id)

    if not files:
        print("No files found in the specified Drive folder.")
        exit(0)

    # Ensure the local download directory exists
    if not os.path.exists(LOCAL_DOWNLOAD_DIR):
        os.makedirs(LOCAL_DOWNLOAD_DIR)
        print(f"Created local directory '{LOCAL_DOWNLOAD_DIR}' for downloads.")

    # Download each file (modify the condition if you want specific files)
    for file in files:
        # Example: Download all files. Modify the condition below if needed.
        # if file['name'] == 'processing_results.txt':  # <--- Modify file name if different
        destination_path = os.path.join(LOCAL_DOWNLOAD_DIR, file['name'])
        print(f"Downloading '{file['name']}'...")
        download_file(service, file['id'], destination_path)

if __name__ == '__main__':
    main()
