# Include Python
FROM python:3.11.1-buster

# Define your working directory
WORKDIR /

# Add your file
COPY . .

# Install runpod
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN python basicsr/setup.py develop

# Call your file when your container starts
CMD [ "python", "deployment.py" ]