
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pylab
import sys
import argparse
import re
from pylab import figure, show, legend, ylabel

from mpl_toolkits.axes_grid1 import host_subplot


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='makes a plot from Caffe output')
  parser.add_argument('caffe_log_file', help='file of caffe captured stdout and stderr')
  parser.add_argument('jpg_file', help='file of save jpg')
  args = parser.parse_args()
  
  f = open(args.caffe_log_file, 'r')

  training_iterations = []
  training_loss = []

  test_iterations = []
  test_accuracy = []
  test_loss = []

  check_test = False
  test_loss_item = 0.0
  for line in f:

    if check_test and 'Test net output #' in line:
      if 'Test net output #0' in line:
        arr = line.split("=")[-1].replace(" loss)", "").strip()
        test_loss_item += float(arr)
      elif 'Test net output #2' in line:
        arr = line.split("=")[-1].replace(" loss)", "").strip()
        test_loss_item += float(arr)
      elif 'Test net output #4' in line:
      #if 'Test net output' in line and 'valid_log_loss' in line:
        arr = line.split("=")[-1].replace(" loss)", "").strip()
        test_loss_item += float(arr)
        test_loss.append(test_loss_item)
        test_loss_item = 0.0
      elif 'Test net output #5' in line:
        test_accuracy.append(float(line.strip().split(' = ')[-1]))
        check_test = False
      #else:
      #  test_loss.append(0)
      #  check_test2 = False
    # training loss
    if '] Iteration ' in line and 'loss = ' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      training_iterations.append(int(arr[0].strip(',')[4:]))
      training_loss.append(float(line.strip().split(' = ')[-1]))
    # testing loss
    if '] Iteration ' in line and 'Testing net' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      test_iterations.append(int(arr[0].strip(',')[4:]))
      check_test = True

  print 'train iterations len: ', len(training_iterations)
  print 'train loss len: ', len(training_loss)
  print 'test loss len: ', len(test_loss)
  print 'test iterations len: ', len(test_iterations)
  print 'test accuracy len: ', len(test_accuracy)

  print test_loss
  if len(test_iterations) != len(test_accuracy): #awaiting test...
    print 'mis-match'
    print len(test_iterations[0:-1])
    test_iterations = test_iterations[0:-1]

  f.close()
#  plt.plot(training_iterations, training_loss, '-', linewidth=2)
#  plt.plot(test_iterations, test_accuracy, '-', linewidth=2)
#  plt.show()
  
  host = host_subplot(111)#, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)

  par1 = host.twinx()

  host.set_xlabel("iterations")
  host.set_ylabel("log loss")
  par1.set_ylabel("validation accuracy")
 
  p1, = host.plot(training_iterations, training_loss, label="training log loss")
  p3, = host.plot(test_iterations, test_loss, label="valdation log loss")
  p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")

  host.legend(loc=5)

  host.axis["left"].label.set_color(p1.get_color())
  par1.axis["right"].label.set_color(p2.get_color())

  plt.draw()
  #plt.show()
  plt.savefig(args.jpg_file, dpi=100)





