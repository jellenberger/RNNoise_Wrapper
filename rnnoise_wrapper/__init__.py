#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or later
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Designed to suppress noise in wav audio using the RNNoise library (https://github.com/xiph/rnnoise).

Contains the RNNoise class. Read more at https://github.com/Desklop/RNNoise_Wrapper.

Dependencies: pydub, numpy.
'''

from .rnnoise_wrapper import RNNoise
