# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# need to be built after quitting conda environment

from tree_sitter import Language

lang_lib = Language.build_library(
    'resource/python.so',
    ["third_party/tree-sitter-python"]
)
print(lang_lib)