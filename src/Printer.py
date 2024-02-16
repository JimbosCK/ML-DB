class Printer:
    def __init__(self, enabled):
        self.enabled = enabled

    def console(self, str, force=False):
        if self.enabled or force:
            print(str)