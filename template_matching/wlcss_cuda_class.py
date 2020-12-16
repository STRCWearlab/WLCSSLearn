import ctypes
import os
from ctypes import *

import numpy as np


class WLCSSCudaParamsTraining:
    """Class for WLCSSCuda computation for parameters training."""

    def __init__(self, templates, streams, num_individuals=1, use_encoding=False):
        """
        Initialization of WLCSSCuda for parameters optimization.
        :param templates: list
            list of templates (numpy.ndarray) with time,data,label,user
        :param streams: list
            list of streams (numpy.ndarray) with time,data,label,user
        :param num_individuals: int
            num of parameters set
        :param use_encoding: boolean
            Use encoding for computing the distance
        """
        if not use_encoding:
            wlcss_dll = ctypes.CDLL(
                os.path.abspath("libs/cuda/training/params/lib_wlcss_cuda_params_training.so"))
        elif use_encoding == '2d':
            wlcss_dll = ctypes.CDLL(
                os.path.abspath("libs/cuda/training/params/lib_wlcss_cuda_params_training_2d_enc.so"))
        elif use_encoding == '3d':
            wlcss_dll = ctypes.CDLL(
                os.path.abspath("libs/cuda/training/params/lib_wlcss_cuda_params_training_3d_enc.so"))
        self._wlcss_init = wlcss_dll.wlcss_cuda_init
        self._wlcss_init.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_int32),
                                    c_int, c_int, c_int, c_int, c_int, c_int]
        self._wlcss_cuda = wlcss_dll.wlcss_cuda
        self._wlcss_cuda.argtype = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int32)]
        self._wlcss_cuda_freemem = wlcss_dll.wlcss_freemem

        self.h_t = templates
        self.h_s = streams

        self.num_templates = len(self.h_t)  # Num block on X
        self.num_streams = len(self.h_s)  # Num block on Y
        self.num_params_sets = num_individuals  # Num thread per block

        self.h_tlen = np.array([len(t) for t in self.h_t]).astype(np.int32)
        self.h_toffsets = np.cumsum(self.h_tlen).astype(np.int32)
        self.h_toffsets = np.insert(self.h_toffsets[0:-1], 0, 0)

        self.h_slen = np.array([len(s) for s in self.h_s]).astype(np.int32)
        self.h_soffsets = np.cumsum(self.h_slen).astype(np.int32)
        self.h_soffsets = np.insert(self.h_soffsets[0:-1], 0, 0)

        # Template as numpy array
        h_ts = np.array([item for sublist in self.h_t for item in sublist]).astype(np.int32)
        # Stream as numpy array
        h_ss = np.array([item for sublist in self.h_s for item in sublist]).astype(np.int32)

        self.h_tmp_windows = np.zeros(
            [(len(h_ts) + 2 * self.num_templates) * self.num_params_sets * self.num_streams]).astype(
            np.int32)
        h_tmp_windows_len = [t + 2 for _ in range(self.num_params_sets) for t in self.h_tlen for _ in
                             range(self.num_streams)]
        # h_tmp_windows_len = np.tile(np.array([t+2 for t in self.h_tlen]), self.num_params_sets * self.num_streams)
        self.h_tmp_windows_offsets = np.cumsum(h_tmp_windows_len).astype(np.int32)
        self.h_tmp_windows_offsets = np.insert(self.h_tmp_windows_offsets[0:-1], 0, 0)

        self.h_mss = np.zeros((len(h_ss) * self.num_params_sets * self.num_templates)).astype(np.int32)
        h_mss_offsets = np.cumsum(np.tile(self.h_slen, self.num_params_sets * self.num_templates)).astype(np.int32)
        self.h_mss_offsets = np.insert(h_mss_offsets, 0, 0)
        self._wlcss_init(self.h_tmp_windows_offsets.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss_offsets.ctypes.data_as(POINTER(c_int32)),
                         h_ts.ctypes.data_as(POINTER(c_int32)), h_ss.ctypes.data_as(POINTER(c_int32)),
                         self.h_tlen.ctypes.data_as(POINTER(c_int32)), self.h_toffsets.ctypes.data_as(POINTER(c_int32)),
                         self.h_slen.ctypes.data_as(POINTER(c_int32)), self.h_soffsets.ctypes.data_as(POINTER(c_int32)),
                         int(self.num_templates), int(self.num_streams), int(self.num_params_sets), int(len(h_ts)),
                         int(len(h_ss)),
                         int(len(self.h_mss)))

    def compute_wlcss(self, parameters):
        """
        Compute the scores between templates and streams, for each parameters set.
        :param parameters: list
            list of parameters set to be evaluated
        :return: list,
            list of numpy.ndarray with the last line of the matching scores between templates and streams
        """
        h_params = np.array(parameters).astype(np.int32)
        self._wlcss_cuda(h_params.ctypes.data_as(POINTER(c_int32)),
                         self.h_mss.ctypes.data_as(POINTER(c_int32)),
                         self.h_tmp_windows.ctypes.data_as(POINTER(c_int32)))
        tmp_mss = np.array([self.h_mss[offset - 1] for offset in self.h_mss_offsets[1:]])
        mss = [np.reshape(np.ravel(x), (self.num_streams, self.num_templates), order='F') for x in
               np.reshape(tmp_mss, (self.num_params_sets, self.num_streams, self.num_templates))]
        return mss

    def cuda_freemem(self):
        """
        Free memory of CUDA
        """
        self._wlcss_cuda_freemem()
