#!/usr/bin/env bash

#build image
IMAGE=registry.datexis.com/tbischoff/enteval

version=0.0.36
echo "Version: $version"
docker build -t $IMAGE -t $IMAGE:$version ../.
docker push $IMAGE:$version
echo "Done pushing image $image for build $version"
