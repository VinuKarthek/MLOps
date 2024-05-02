# Use Python 3.9 base image
FROM python:3.9

# Set working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Streamlit Python file
COPY main.py .

# Define an environment variable for the start command (default is streamlit command)
ENV SERVER_PORT = "8080"
ENV START_COMMAND="uvicorn main:app --host 0.0.0.0 --port $SERVER_PORT --reload"
# uvicorn main:app --host 0.0.0.0 --port $SERVER_PORT --workers 8 --header server:custom-server-name

# Set the command to start the Streamlit app using the environment variable
CMD ["/bin/sh", "-c", "$START_COMMAND"]
