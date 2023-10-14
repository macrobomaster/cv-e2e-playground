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
              nevergrad = p.buildPythonPackage rec {
                pname = "nevergrad";
                version = "0.14.0";
                src = pkgs.fetchFromGitHub {
                  owner = "facebookresearch";
                  repo = pname;
                  rev = version;
                  sha256 = "sha256-qHcrpyc9/pPyQUrZTq1A3KpAn16VlpDi+wTdOVb726I=";
                };
                doCheck = false;
                propagatedBuildInputs = with p; [
                  bayesian-optimization
                  cma
                  numpy
                  pandas
                  typing-extensions
                ];
              };
              tinygrad = p.buildPythonPackage {
                pname = "tinygrad";
                version = inputs.tinygrad.shortRev;
                src = inputs.tinygrad;
                doCheck = false;
                propagatedBuildInputs = with p; [
                  networkx
                  nevergrad
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
