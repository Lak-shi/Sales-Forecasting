from google.cloud import storage

# Google Cloud Storage Config
# BUCKET_NAME = ""
# SOURCE_FILE = ""
# DESTINATION_BLOB = ""

def upload_to_gcs(bucket_name, source_file, destination_blob):
    """Uploads a file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)

    blob.upload_from_filename(source_file)
    print(f"✅ {source_file} uploaded to gs://{bucket_name}/{destination_blob}")

# Run the upload
upload_to_gcs(BUCKET_NAME, SOURCE_FILE, DESTINATION_BLOB)

