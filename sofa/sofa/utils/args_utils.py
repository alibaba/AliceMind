import argparse
import base64
import json
import copy

class ArgsBase(object):
    _default_parser = argparse.ArgumentParser(description='PyTorch BERT Model')

    def __init__(self, parser=None):
        self.parser = parser
        self.args = None
    
    @staticmethod
    def _get_default_parser():
        return ArgsBase._default_parser
    
    @staticmethod
    def decode_parameters(parameters):
        parameters = base64.b64decode(parameters).decode('utf-8')
        return json.loads(parameters)

    @staticmethod
    def check_and_reset_params(params, information):
        new_params = copy.copy(params)
        for name, func, required, default_value in information:
            if required and name not in params:
                assert name in params
            if name in params:
              new_params[name] = func(params[name])
            else:
                new_params[name] = default_value
        return new_params

    @staticmethod
    def check_and_reset_parameters(params, information):
        new_params = copy.copy(params)
        for name, func, required, default_value in information:
            if required and name not in params:
                assert name in params
            if name in params:
                new_params[name] = func(params[name])
            else:
                new_params[name] = default_value
        return new_params    
    
    def _merge_args(self, args=None):
        if args is None:
            return self.args
        """ merge args with """
        local_keys = self.args.__dict__.keys()
        for key, value in args.__dict__.items():
            if key in local_keys:
                continue
            self.args.__dict__[key] = value
        return self.args
   
    def _parse_args(self):
        """new parser should create a parse_args method"""
        return self.parser.parse_args()
    
    def parse_args(self, args=None, replace=False):
        if replace and args:
            return args
        self.args = self._parse_args()
        self.args = self._merge_args(args)        
        return self.args
    
    
    