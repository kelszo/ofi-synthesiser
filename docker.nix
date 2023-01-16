{ pkgs ? import <nixpkgs> { } }:

let
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix";
    ref = "refs/tags/3.5.0";
  }) { };

  py = mach-nix.mkPython {
    python = "python39";
    requirements = builtins.readFile ./requirements.txt;
  };

  nvidia_x11 = pkgs.linuxKernel.packages.linux_5_10.nvidia_x11;

  img = pkgs.dockerTools.buildImage {
    name = "ofi-synthesiser";
    tag = "latest";
    copyToRoot = pkgs.buildEnv {
      name = "image-root";
      paths = [ ./. py pkgs.cudatoolkit nvidia_x11 ];
      pathsToLink = [ "/bin" "/ofisynthesiser" "/data" ];
    };
    config = {
      Env = [ ''LD_LIBRARY_PATH = "${nvidia_x11}/lib"'' "PATH=/bin:$PATH" ];
      Cmd = [ "python" "-m" "ofisynthesiser.executors.run" ];
    };
  };
in img
