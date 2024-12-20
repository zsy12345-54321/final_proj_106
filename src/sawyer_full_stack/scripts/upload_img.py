import os
import mimetypes
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Define constants for paths and folder names
IMAGES_SOURCE_DIR = ''  # <--- Modify this path to your images directory
DRIVE_UPLOAD_FOLDER_NAME = 'Uploaded_Images'  # <--- Modify the folder name in Google Drive if needed
CREDENTIALS_FILE = 'final_proj.json'  # Ensure this file is in the same directory as the script
TOKEN_FILE = 'token.json'  # Token file for authentication

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate():
    """Authenticate the user and return the Drive service."""
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
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
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    return service

def create_drive_folder(service, folder_name):
    """Create a folder in Google Drive and return its ID."""
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    try:
        folder = service.files().create(body=file_metadata, fields='id').execute()
        print(f"Created folder '{folder_name}' with ID: {folder.get('id')}")
        return folder.get('id')
    except Exception as e:
        print(f"Failed to create folder '{folder_name}'. Error: {e}")
        exit(1)

def get_drive_folder_id(service, folder_name):
    """Retrieve the folder ID if it exists, else create it."""
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if items:
        folder_id = items[0]['id']
        print(f"Found existing folder '{folder_name}' with ID: {folder_id}")
        return folder_id
    else:
        return create_drive_folder(service, folder_name)

def file_exists(service, folder_id, file_name):
    """Check if a file already exists in the folder with the same name."""
    query = f"'{folder_id}' in parents and name = '{file_name}' and trashed = false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if items:
        return items[0]['id']  # Return the existing file ID
    return None

def upload_file(service, file_path, parent_folder_id=None):
    """Upload a single file to Google Drive."""
    file_name = os.path.basename(file_path)
    file_id = file_exists(service, parent_folder_id, file_name)
    
    if file_id:
        # If the file exists, delete the existing file
        service.files().delete(fileId=file_id).execute()
        print(f"Deleted existing file '{file_name}' in Drive.")
    
    # Upload the new file
    file_metadata = {'name': file_name}
    if parent_folder_id:
        file_metadata['parents'] = [parent_folder_id]
    
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
    try:
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"Uploaded {file_path} with File ID: {file.get('id')}")
    except Exception as e:
        print(f"Failed to upload {file_path}. Error: {e}")

def upload_images(service, source_dir, parent_folder_id=None):
    supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    source_path = Path(source_dir)
    print(source_path)
    if not source_path.is_dir():
        print(f"The source directory {source_dir} does not exist or is not a directory.")
        exit(1)

    for file_path in source_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            upload_file(service, str(file_path), parent_folder_id)

def main():
    # Authenticate and get the Drive service
    service = authenticate()

    # Get or create the upload folder in Google Drive
    folder_id = get_drive_folder_id(service, DRIVE_UPLOAD_FOLDER_NAME)

    # Upload images from the source directory to Google Drive
    upload_images(service, IMAGES_SOURCE_DIR, folder_id)

if __name__ == '__main__':
    main()

