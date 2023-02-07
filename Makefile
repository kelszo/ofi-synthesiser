podman_build:
	nix-build -o ./local/docker-image.tar.gz 

podman_cuda_build:
	podman build -f Containerfile -t ofi-synthesiser:latest

podman_cuda_save:
	podman save ofi-synthesiser:latest | gzip > local/docker-image-cuda.tar.gz

podman_load:
	podman load < ./local/docker-image-cuda.tar

podman_run:
	podman run --name ofi-synthesiser --rm --volume=$(pwd)/out:/out localhost/ofi-synthesiser:latest 

docker_run_cuda:
	nvidia-docker run --name ofi-synthesiser -d  --gpus all --volume=$(pwd)/ofisynthesiser:/app/ofisynthesiser --volume=$(pwd)/out:/app/out ofi-synthesiser:latest python -m ofisynthesiser.executors.run
