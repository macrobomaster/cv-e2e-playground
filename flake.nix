{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    tinygrad.url = "github:wozeparrot/tinygrad-nix/9dc15f3f8dd2021522690ad514967e7589d77503";
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
              graphviz
              llvmPackages_latest.clang
            ];

          shellHook = ''
            export HIP_PATH="${pkgs.rocmPackages.clr}"
          '';
        };
      }
    );
}
