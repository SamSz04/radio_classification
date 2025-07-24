import sys


class Logger(object):
    def __init__(self, file_name="Default.log"):
        self.terminal = sys.stdout
        self.log = open(file_name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush() 

    def flush(self):
        self.log.flush()
