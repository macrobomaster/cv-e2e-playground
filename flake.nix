{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    tinygrad.url = "github:wozeparrot/tinygrad-nix";
  };

  outputs = inputs @ {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem
    (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShell = pkgs.mkShell {
          packages = let
            python-packages = p:
              with p; [
                pydot
                inputs.tinygrad.packages.${system}.default
                torch
                (opencv4.override {
                  enableGtk3 = true;
                })
                wandb
                onnx
                onnxruntime
                (onnxconverter-common.override {
                  protobuf = protobuf;
                })
                llvmlite
              ];
            python = pkgs.python311;
          in
            with pkgs; [
              (python.withPackages python-packages)

              # needed for GRAPH=1 to work
              graphviz
            ];
        };
      }
    );
}
