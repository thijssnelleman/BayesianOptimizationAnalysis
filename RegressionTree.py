#Aaron Kannangara, s1366106, University Leiden, Liacs.
#This program can be called and is used to run
#GP and randomsearch on a given file name.
#The results are published in a plot if the flag is set.

#FLAGS:
#-f to define dataset file name, name should be provided after -f
#-d second fileinput
#-i third fileinput
#-e is to run multiple experiments and take average and find standard deviation
#-g is used to run guassian process experiment
#-r used to run random search experiment
#-n used to define nr_calls
#-s will save the plot


import sys
import argparse
import numpy as np

from GuassianProcess import GaussianProcess
from RandomSearch import RandomSearch
import matplotlib.pyplot as plt


def convergenceStdGroups(groupings, nr_calls):
	res_g = []
	#groups results per call together
	for n in range(nr_calls):
		res_n = []
		for g in groupings:
			res_n.append(g[n])
		res_g.append(res_n)
		del res_n
	#find average and standard deviation for each n
	res = []
	std = []
	for r in res_g:
		std.append(np.std(r))
		res.append(np.mean(r))
	return res, std


def addToPlot(y, std, label):
	
	if std == None:
		plt.plot(y, label=label)
	else:
		data = {
			'x': list(range(1, len(y) + 1)),
			'y1': [Y - e for Y, e, in zip(y,std)],
			'y2': [Y + e for Y, e, in zip(y,std)]
		}
		x=np.arange(1, len(y) + 1, step=1)
		plt.errorbar(x=x, y=y, label=label, yerr=std)
		plt.fill_between(**data, alpha=0.25)


def makePlot(save, name):
	plt.legend(loc="best")
	plt.xlabel("Function evaluations")
	plt.ylabel("Min F(x)")
	plt.title("")
	plt.grid(which='both', axis='y')
	#plt.ylim(None,0)
	if save:
		plt.savefig(name)
	else:
		plt.show()


def runExperiments(args):
	if args.RS:
		RS = RandomSearch(args.filename, args.nr_calls)
		if args.experiment:
			RS_res = RS.run_groups()
			mean, std = convergenceStdGroups(RS_res, args.nr_calls)
			addToPlot(mean, std, "Random Search")
		else:
			RS_res = RS.run()
			addToPlot(RS_res, None, "Random Search")
	if args.GP:
		GP = GaussianProcess(args.filename, args.nr_calls)
		if args.experiment:
			GP_res = GP.run_groups()
			mean, std = convergenceStdGroups(GP_res, args.nr_calls)
			addToPlot(mean, std, "GP")
		else:
			GP_res = GP.run()
			addToPlot(GP_res, None, "GP")
	if args.plot:
		name = "results/"+str(args.nr_calls)+"_"
		if args.GP:
			name+="GP_"
		if args.RS:
			name+="RS_"
		name+=args.filename[:-4]+".jpg"
		makePlot(args.save, name)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--filename", required=True)
	parser.add_argument("-d", "--filename2", required=False)
	parser.add_argument("-i", "--filename3", required=False)
	parser.add_argument("-g", "--GP", action="store_true", default=False)
	parser.add_argument("-r", "--RS", action="store_true", default=False)
	parser.add_argument("-p","--plot", action="store_true", default=False)
	parser.add_argument("-e","--experiment", action="store_true",
						default=False)
	parser.add_argument("-s","--save", action="store_true", default=False)
	parser.add_argument("-n","--nr_calls", required=True, type=int)
	args = parser.parse_args()
	if (not(args.GP) and not(args.RS)):
		print("Error: no test has been specified")
		exit()
	runExperiments(args)
