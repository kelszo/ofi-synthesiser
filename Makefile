docker_build:
	nix-build ./docker.nix -o ./local/docker-image

docker_load:
	podman load < ./local/docker-image

docker_run:
	podman run --rm localhost/ofi-synthesiser:latest --volume=./out:/out