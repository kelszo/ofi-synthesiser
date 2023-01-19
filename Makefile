docker_build:
	nix-build ./docker.nix -o ./local/docker-image.tar.gz 

docker_cuda_build:
	nix-build ./docker.nix -o ./local/docker-cuda-image.tar.gz --arg cuda true

docker_load:
	podman load < ./local/docker-image.tar.gz

docker_run:
	podman run --name ofi-synthesiser --rm --volume=$(pwd)/out:/out localhost/ofi-synthesiser:latest 

docker_run_cuda:
	nvidia-docker run --name ofi-synthesiser -d --rm  --gpus all --volume=$(pwd)/out:/out ofi-synthesiser:cuda
