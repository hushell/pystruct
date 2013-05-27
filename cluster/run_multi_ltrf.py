import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from pystruct.models import LatentTRF
#from pystruct.learners import LatentSSVM
from pystruct.learners import LatentSubgradientSSVM
import scipy.io
import os
import time

tot_classes = 15
n_machines = tot_classes * (tot_classes - 1) / 2 # c^n_2

#import ipdb
#ipdb.set_trace()

#os.system('source /scratch/a1/sge/settings.sh')
os.system('rm -f log/*')

name_data = '15sc_'
bash_del = False

for i in xrange(tot_classes):
    for j in xrange(i+1, tot_classes):
        print '------------ training machine (%d,%d) -------------' % (i+1,j+1)

        script = \
            '#!/bin/bash' + '\n' \
            + '#' + '\n' \
            + '# use current working directory for input and output - defaults is' + '\n' \
            + '# to use the users home directory' + '\n' \
            + '#$ -cwd' + '\n' \
            + '#' + '\n' \
            + '# name this job' + '\n' \
            + '#$ -N ltrf_%s_c%d%d' % (name_data,i+1,j+1) + '\n' \
            + '#' + '\n' \
            + '# send stdout and stderror to this file' + '\n' \
            + '#$ -o log/%sltrf_c%d%d.out' % (name_data,i+1,j+1) + '\n' \
            + '#$ -j y' + '\n\n' \
            + '#see where the job is being run' + '\n' \
            + 'hostname' + '\n\n' \
            + 'export PATH=/nfs/stak/students/h/huxu/python-cluster/bin:$PATH' + '\n' \
            + 'export INCLUDE_PATH=$INCLUDE_PATH:/nfs/stak/students/h/huxu/python-cluster/include' + '\n' \
            + 'export LIBRARY_PATH=$LIBRARY_PATH:/nfs/stak/students/h/huxu/python-cluster/lib' + '\n' \
            + 'export LD_LIBRARY_PATH=/nfs/stak/students/h/huxu/python-cluster/lib:$LD_LIBRARY_PATH' + '\n' \
            + 'export PYTHONPATH=/nfs/stak/students/h/huxu/python-cluster/bin:/nfs/stak/students/h/huxu/python-cluster/git-dir:$PYTHONPATH' + '\n\n' \
            + '# print date and time' + '\n' \
            + 'date' + '\n' \
            + '# python code' + '\n' \
            + 'python multi_ltrf_par.py %sdata/15sc_c%02d.mat ' % (name_data, i+1) + '%sdata/15sc_c%02d.mat ' % (name_data, j+1) + 'store/%sclf_c%d%d.p ' % (name_data,i+1,j+1) + '4 2 200' + '\n' \
            + '# print date and time again' + '\n' \
            + 'date' + '\n'

        bash_file = open('sge_%sltrf_c%d%d.sh' % (name_data,i+1,j+1), 'w')
        bash_file.write(script)
        bash_file.close()

        os.system('chmod +x ' + 'sge_%sltrf_c%d%d.sh' % (name_data,i+1,j+1))
        #os.system('sh ' + 'sge_%sltrf_c%d%d.sh' % (name_data,i+1,j+1))
        os.system('qsub ' + 'sge_%sltrf_c%d%d.sh' % (name_data,i+1,j+1))
        time.sleep(1.13)
        if bash_del == True:
            os.system('rm -f ' + 'sge_%sltrf_c%d%d.sh' % (name_data,i+1,j+1))

