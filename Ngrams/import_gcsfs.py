import gcsfs

# Initialize the GCS file system
fs = gcsfs.GCSFileSystem('transformer-ngrams')

# Define the path to the model on GCS
model_path = 'gs://transformer-ngrams/32768.model'

# Download the model to your local file system
local_model_path = '32768.model'  # Change to your desired local path

with fs.open(model_path, 'rb') as fsrc:
    with open(local_model_path, 'wb') as fdst:
        fdst.write(fsrc.read())

print(f"Model downloaded to: {local_model_path}")
