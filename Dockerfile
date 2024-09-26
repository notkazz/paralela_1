# Use uma imagem base com GCC e suporte ao OpenMP
FROM gcc:latest

# Definir o diretório de trabalho
WORKDIR /usr/src/app

# Copiar o código fonte para o diretório de trabalho
COPY bellman_ford_parallel.c .

# Compilar o código fonte com suporte ao OpenMP
RUN gcc -o bellman_ford_parallel bellman_ford_parallel.c -fopenmp

# Definir o comando padrão ao executar o contêiner
CMD ["./bellman_ford_parallel"]