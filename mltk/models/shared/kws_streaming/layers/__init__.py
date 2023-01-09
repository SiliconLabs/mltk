# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# kws_streaming.layers
import os
import sys

mltk_models_shared_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
if mltk_models_shared_dir not in sys.path:
    sys.path.append(mltk_models_shared_dir)