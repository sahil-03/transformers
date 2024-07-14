cmake_minimum_required(VERSION 3.10)
project(transformers)

add_executable(transformer components/attention.cpp components/attention.h utils/tensor_operations.h)