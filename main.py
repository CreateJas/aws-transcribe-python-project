import streamlit as st
import asyncio
import sounddevice
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import uuid
import time
import json
import requests
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# Initialize AWS service clients with your specific region
region_name = 'ap-south-1'  # Mumbai
s3_client = boto3.client('s3', region_name=region_name)
transcribe_client = boto3.client('transcribe', region_name=region_name)

class MyEventHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                st.write(alt.transcript)

def create_bucket(bucket_name, region):
    """Create an S3 bucket in a specified region"""
    try:
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
        st.success(f"Bucket '{bucket_name}' created successfully in the {region} region.")
        return True
    except ClientError as e:
        st.error(e)
        return False
    except NoCredentialsError:
        st.error("Credentials not available")
        return False

def upload_file_to_bucket(bucket_name, file):
    """Upload a file to an S3 bucket"""
    object_name = file.name
    try:
        s3_client.upload_fileobj(file, bucket_name, object_name)
        st.success(f"File '{file.name}' uploaded successfully to bucket '{bucket_name}'.")
        return f"s3://{bucket_name}/{object_name}"
    except ClientError as e:
        st.error(e)
        return None

def start_transcription(bucket_name, file_uri):
    """Start transcription job"""
    job_name = f"transcription-{uuid.uuid4()}"
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': file_uri},
            MediaFormat=file_uri.split('.')[-1],
            LanguageCode='en-US',
        )
        return job_name
    except ClientError as e:
        st.error(e)
        return None

def get_transcription_result(job_name):
    """Check the status of the transcription job and return the result"""
    try:
        response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = response['TranscriptionJob']['TranscriptionJobStatus']
        if status == 'COMPLETED':
            transcript_file_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
            transcript_response = requests.get(transcript_file_uri)
            transcript_data = transcript_response.json()
            return transcript_data['results']['transcripts'][0]['transcript']
        elif status == 'FAILED':
            st.error("Transcription job failed.")
            return None
        else:
            return "IN_PROGRESS"
    except ClientError as e:
        st.error(e)
        return None

async def mic_stream():
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
        asyncio.run_coroutine_threadsafe(input_queue.put((bytes(indata), status)), loop)

    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=16000,
        callback=callback,
        blocksize=1024 * 2,
        dtype="int16",
    )
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status

async def write_chunks(stream):
    async for chunk, status in mic_stream():
        await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()

async def basic_transcribe():
    client = TranscribeStreamingClient(region="us-east-1")

    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm"
    )

    handler = MyEventHandler(stream.output_stream)
    await asyncio.gather(write_chunks(stream), handler.handle_events())

st.title('Audio Transcription')

# UI to create a new S3 bucket
bucket_name = st.text_input('Enter a name for a new S3 bucket or an existing one:')
if st.button('Create/Use Bucket'):
    create_bucket(bucket_name, region_name)

# UI to upload a file
file = st.file_uploader("Upload audio file (.wav or .mp3)", type=['wav', 'mp3'])
if file is not None and bucket_name:
    file_uri = upload_file_to_bucket(bucket_name, file)
    if file_uri:
        job_name = start_transcription(bucket_name, file_uri)
        if job_name:
            st.write("Transcription job started. This might take a while...")
            transcript_status = "IN_PROGRESS"
            while transcript_status == "IN_PROGRESS":
                transcript_status = get_transcription_result(job_name)
                if transcript_status not in ["IN_PROGRESS", None]:
                    st.write("Transcription Completed:")
                    st.write(transcript_status)
                    break
                elif transcript_status is None:
                    break
                time.sleep(5)  # Poll every 5 seconds
else:
    st.write("Or start live transcription:")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_debug(True)

    loop.run_until_complete(basic_transcribe())
