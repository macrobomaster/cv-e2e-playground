{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
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
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            inputs.tinygrad.overlays.default
          ];
        };
      in {
        devShell = pkgs.mkShell {
          packages = let
            python-packages = p:
              with p; [
                pydot
                tinygrad
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
