docker build -t bellman_ford_parallel .

docker run -it --cpuset-cpus="0-15" --rm bellman_ford_parallel