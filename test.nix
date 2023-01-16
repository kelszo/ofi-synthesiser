{ pkgs ? import <nixpkgs> { } }:
let
  myAppEnv = pkgs.poetry2nix.mkPoetryEnv {
    projectDir = ./.;
    python = pkgs.python39;
    editablePackageSources = { my-app = ./ofisynthesiser; };
    preferWheels = true;
  };
in myAppEnv.env
