# Use an official Ubuntu as the base image
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Set the working directory inside the container
WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    wget \
    unzip \
    libopencv-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    build-essential \
    && apt-get clean

# Copy the Haar Cascade classifier XML file into the container
COPY haarcascade_frontalface_default.xml /app/

# Copy the source code (app.c) into the container
COPY app.c /app/

# Copy the images folder into the container
COPY Imagens /app/Imagens

# Compile the face detection C++ program with OpenMP support
RUN g++ -fopenmp -std=c++17 app.c -o app `pkg-config --cflags --libs opencv4`

# Set the default command to run the compiled app
CMD ["./app"]
