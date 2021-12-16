# Written by ☘ Molpaxbio Co., Ltd.
# Author: jaesik.won@molpax.com
# Version: 0.5

from pathlib import Path

class Logger:
    def __init__(self, path, name, mode='a'):
        assert mode == 'a' or mode == 'w'
        path = Path(path)
        
        self.setname(name)
        self.__f = open(path / (name + '_log.txt'), mode)
        self.__p = '%-10s' + '%-10s' + '%s'
        self.__verbose = True
        self.log("Logger opened")
        s = "append" if mode == 'a' else "overwrite"
        self.log(f"Log will be saved at {path / (name + '_log.txt')}, mode {s}")
    
    # log function
    def log(self, msg):
        if self.__verbose:
            s = self.__p % (self.__name, '[Log]', msg)
            print(s)
            self.__f.write(s + '\n')
        
    def process(self, msg):
        if self.__verbose:
            s = self.__p % (self.__name, '[Process]', msg)
            print(s)
            self.__f.write(s + '\n')
    
    def result(self, msg):
        if self.__verbose:
            s = self.__p % (self.__name, '[Result]', msg)
            print(s)
            self.__f.write(s + '\n')
    
    # setter / getter  
    def setverbose(self, b:bool):
        self.__verbose = b
        
    def setname(self, name):
        self.__name = '[' + name + ']'
    
    def getname(self):
        return self.__name[1:-1]
    
    # destructor
    def close(self):
        self.log("Logger closed.")
        self.__f.close()
        self.__f = None
    
    def __del__(self):
        if self.__f is not None:
            self.close()