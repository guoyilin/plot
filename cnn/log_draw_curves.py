import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import sys
import math

if len(sys.argv) < 2:
	print "Usage: python log_draw_curves.py log_filename [loss_y_min] [loss_y_max]"
	exit()

filename = sys.argv[1]

# open log file
log_file = open(filename)
lines = log_file.readlines()
n = len(lines)

# define list
train_loss_x = []
train_loss_y = []
validate_loss_x = []
validate_loss_y = []
accuracy_x = []
accuracy_y = []

# initialize test loss
validate_loss = 0.0
for i in range(0,n):
	if "loss =" in lines[i]:
		train_loss_x.append(int(lines[i].split()[5].split(",")[0]))
		train_loss_y.append(float(lines[i].split()[8]))
	elif "Testing net" in lines[i]:
		iteration = int(lines[i].split()[5].split(",")[0])
		validate_loss_x.append(iteration)
		accuracy_x.append(iteration)
	elif "Test net output" in lines[i]:
		if "* 0.3" in lines[i] or "* 1" in lines[i]:
			validate_loss += float(lines[i].split()[14])
		elif "loss3/top-1 =" in lines[i]:		
			validate_loss_y.append(validate_loss)
			validate_loss = 0.0
			accuracy_y.append(float(lines[i].split()[10]))

plt.figure()
if len(accuracy_y) != 0:
	ax1 = plt.subplot(211) 
	ax2 = plt.subplot(212)
	plt.subplots_adjust(hspace=0.4)
else:
	ax1 = plt.gca()

# plot loss
if len(sys.argv) == 4:
	loss_miny = float(sys.argv[2])
	loss_maxy = float(sys.argv[3])
else:
	if len(validate_loss_y) != 0:
		loss_miny = min([min(train_loss_y),min(validate_loss_y)])
		loss_maxy = max([max(train_loss_y), max(validate_loss_y)])
	else:
		loss_miny = min(train_loss_y)
		loss_maxy = max(train_loss_y)
	loss_miny = math.floor(float(loss_miny)*100)/100.0
	loss_maxy = math.ceil(float(loss_maxy)*100)/100.0

loss_minx = int(min(train_loss_x))
loss_maxx = int(max(train_loss_x))/10000*10000

plt.sca(ax1)
plt.plot(train_loss_x,train_loss_y,label="train_loss",color="blue",linewidth=1)
if len(validate_loss_y) != 0:
	plt.plot(validate_loss_x,validate_loss_y,label="validate_loss",color="green",linewidth=1)
plt.xlabel("Iteration", fontsize=10)
plt.ylabel("Loss", fontsize=10)
plt.title("Loss", fontsize=12)
plt.ylim(loss_miny,loss_maxy)
plt.xlim(loss_minx,loss_maxx)
#plt.ylim(0, 1)
ax = plt.gca()
ax.xaxis.set_major_locator( MultipleLocator( (loss_maxx-loss_minx)/10 ) )
ax.yaxis.set_major_locator( MultipleLocator( (loss_maxy-loss_miny)/10 ) )
#ax.yaxis.set_major_locator( MultipleLocator(0.1) )
plt.grid()
#plt.legend(loc=0, fontsize=8)
plt.legend(loc=0, prop={'size':8})

# set axis fontsize
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(7)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(7)

if len(accuracy_y) != 0:
# plot accuracy
	accuracy_miny = math.floor(float(min(accuracy_y))*100)/100.0
	accuracy_maxy = math.ceil(float(max(accuracy_y))*100)/100.0
	accuracy_minx = int(min(accuracy_x))
	accuracy_maxx = int(max(accuracy_x))/10000*10000

	
	plt.sca(ax2)
	plt.plot(accuracy_x, accuracy_y, label="accuracy", color="red", linewidth=1)
	plt.xlabel("Iteration", fontsize=10)
	plt.ylabel("Accuracy", fontsize=10)
	plt.title("Validate Accuracy", fontsize=12)
	plt.ylim(accuracy_miny,accuracy_maxy)
	plt.xlim(accuracy_minx,accuracy_maxx)
	ax = plt.gca()
	ax.xaxis.set_major_locator( MultipleLocator( ( ( loss_maxx-loss_minx )/10 ) ) )
	ax.yaxis.set_major_locator( MultipleLocator( ( ( accuracy_maxy-accuracy_miny )/10 ) ) )
	plt.grid()
	
	# set axis fontsize
	for tick in ax.xaxis.get_major_ticks():
	    tick.label1.set_fontsize(7)
	for tick in ax.yaxis.get_major_ticks():
	    tick.label1.set_fontsize(7)

plt.savefig(filename+".jpg", dpi=200)
