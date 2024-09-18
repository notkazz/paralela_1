docker build -t face-detection .

docker run -v /caminho/local/Imagens:/app/Imagens face-detection
docker run -v /C/paralela_1/Imagens:/app/Imagens face-detection