{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    tinygrad = {
      url = "github:tinygrad/tinygrad";
      flake = false;
    };
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
            python-packages = p: let
              tinygrad = p.buildPythonPackage {
                pname = "tinygrad";
                version = inputs.tinygrad.shortRev;
                src = inputs.tinygrad;
                doCheck = false;
                propagatedBuildInputs = with p; [
                  networkx
                  numpy
                  pillow
                  pyopencl
                  pyyaml
                  requests
                  tqdm
                ];
              };
            in
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
