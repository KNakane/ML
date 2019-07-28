import os, sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Utility():
    def __init__(self, prefix=None):
        dt_now = datetime.datetime.now()
        self.res_dir = "results/"+dt_now.strftime("%y%m%d_%H%M%S")
        if prefix is not None:
            self.res_dir = self.res_dir + "_{}".format(prefix)
        self.log_dir = self.res_dir + "/log"
        self.tf_board = self.res_dir + "/tf_board"
        self.model_path = self.res_dir + "/model"
        self.saved_model_path = self.model_path + "/saved_model"

    def initialize(self):
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        return

    def write_configuration(self, message, _print=False):
        """
        設定をテキストに出力する
        parameters
        -------
        message : dict
        _print : True / False : terminalに表示するか
        """
        with open(self.log_dir + '/log.txt', 'a') as f:
            f.write("------Learning Details------\n")
            if _print:
                print("------Learning Details------")
            for key, info in message.items():
                f.write("%s : %s\n"%(key, info))
                if _print:
                    print("%s : %s"%(key, info))
            f.write("----------------------------\n")
            print("----------------------------")
        return 