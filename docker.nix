{ pkgs ? import <nixpkgs> {
  config.allowUnfree = true;
  config.cudaSupport = true;
  config.cudnnSupport = true;
}, lib ? pkgs.lib, cuda ? false }:

let
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix";
    ref = "refs/tags/3.5.0";
  }) { };

  py = mach-nix.mkPython {
    python = "python39";
    requirements = builtins.readFile ./requirements.txt;
  };

  img = pkgs.dockerTools.buildLayeredImage {
    name = "ofi-synthesiser";
    tag = if cuda ? true then "cuda" else "latest";
    contents = [ ./. py pkgs.busybox ]
      ++ lib.optionals cuda [ pkgs.cudatoolkit ];

    config = {
      Env = [ "PATH=/bin:$PATH" ] ++ lib.optionals cuda [
        "LD_LIBRARY_PATH=/usr/lib64"
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
        "NVIDIA_VISIBLE_DEVICES=all"
      ];
      Cmd = [ "python" "-m" "ofisynthesiser.executors.run" ];
    };
  };
in img
