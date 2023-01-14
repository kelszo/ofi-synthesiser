{
  description = "ofi-synthesiser";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    mach-nix.url = "mach-nix/3.5.0";
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonEnv = mach-nix.lib."${system}".mkPython {
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
      in { devShells.default = import ./shell.nix { inherit pkgs; }; });
}
