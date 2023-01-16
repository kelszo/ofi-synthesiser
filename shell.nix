{ pkgs ? import <nixpkgs> { } }:
let
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix";
    ref = "refs/tags/3.5.0";
  }) { };
  pythonEnv =
    mach-nix.mkPython { requirements = builtins.readFile ./requirements.txt; };
in pkgs.mkShell { buildInputs = [ pythonEnv ]; }
