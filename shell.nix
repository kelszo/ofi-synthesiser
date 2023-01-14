{ pkgs ? import <nixpkgs> { } }:
let
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix";
    ref = "refs/tags/3.5.0";
  }) { };
  pythonEnv = mach-nix.mkPython {
    requirements = builtins.readFile ./requirements.txt + ''
      black
      python-lsp-server
      pyls-isort
      jupyter_core
      nbconvert

      #python310Packages.pyls-black
      #python310Packages.pylsp-mypy
    '';
  };
in pkgs.mkShell {
  # buildInputs is for dependencies you'd need "at run time",
  # were you to to use nix-build not nix-shell and build whatever you were working on
  buildInputs = [ pythonEnv ];
}
