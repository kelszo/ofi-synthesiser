{
  description = "ofi-synthesiser";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pypi-deps-db = {
      url = "github:DavHau/pypi-deps-db";
      flake = false;
    };
    mach-nix = {
      url = "mach-nix/3.5.0";
      inputs.pypi-deps-db.follows = "pypi-deps-db";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix, pypi-deps-db }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            config.allowUnfree = true;
            config.cudaSupport = true;
            config.cudnnSupport = true;
          };
        };

        pythonEnv = mach-nix.lib."${system}".mkPython {
          python = "python38";
          requirements = builtins.readFile ./requirements.txt;
        };

        nvidiaPytorch = pkgs.dockerTools.pullImage {
          imageName = "nvcr.io/nvidia/pytorch";
          finalImageName = "nvcr.io/nvidia/pytorch";
          finalImageTag = "23.01-py3";
          imageDigest =
            "sha256:cbaa53e58a9f0aa8510fda7fba9e29ef5f14ca3ada280ce2ab601881a3cd9618";
          sha256 = "4VH2wzv6gsORd54yyP3dfmRv/6RF7/8ie830tzXxCIk=";
        };

        dockerImage = pkgs.dockerTools.buildImage {
          name = "ofi-synthesiser";
          fromImage = nvidiaPytorch;
          tag = "latest";
          created = "now";
          contents = [ ./. ];
          runAsRoot = ''
            pip install --upgrade --no-cache-dir pip \
              && pip install --upgrade --no-cache-dir \
              catboost \
              lightgbm \
              optuna \
              sdv 
          '';

          config = {
            Env = [
              "PATH=/bin:$PATH"
              "LD_LIBRARY_PATH=/usr/lib64"
              "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
              "NVIDIA_VISIBLE_DEVICES=all"
            ];
            Cmd = [ "/bin/bash" ];
          };
        };
      in {
        packages = { docker = dockerImage; };

        defaultPackage = dockerImage;

        devShell = pkgs.mkShell {
          nativeBuildInputs = [ ];
          buildInputs = [
            pythonEnv
            pkgs.pandoc
            pkgs.haskellPackages.pandoc-crossref
            pkgs.graphviz
          ];
          shellHook = ''
            echo ""
            echo "Current python: $(which python)"
          '';

        };
      });
}
