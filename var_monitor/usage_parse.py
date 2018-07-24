'''
Created on Jan 19, 2018

@author: Francesc Torradeflot
'''

import glob
import logging
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import marshal
import tempfile
import io
import pstats

logger = logging.getLogger(__file__)

conversion_dict = {'K': -2, 'M': -1, 'G': 0}
def conversion(x):
    """ Converts a given number (x) into a more readable format """
    return x[-1] in conversion_dict and float(x[:-1])*1024.**conversion_dict[x[-1]] or 0.0

def save_or_show(fig, save_plot=False, plot_file=None):
    """ Saves or shows a plot.

        Parameters
        ----------
        fig : IDK________________________________________________
            plot to be saved or shown.
        save_plot : bool
                - True : Plot is saved into a .pdf file.
                - False : Plot is shown but not saved
        plot_file : ________________________________________________
            Name of the file where the plot will be saved.

        Raise
        -----


    """

    print ("SAVE PLOT : ", type(fig))
    if save_plot:
        if plot_file is None:
            raise Exception('File not informed')

        fig.savefig(plot_file)
    else:
        plt.show()

def compute_df_columns(df):
    """ Creates the columns for a given log file storing the data
        in a convenient format

        Parameters
        ----------
        df : IDK________________________________________________
            plot to be saved or shown.
    """

    #print ("COMPUTE DF COLUMNS : ", df, type(df))
    if len(df['timestamp']) == 0:
        return None

    df['timestamp'] = df['timestamp'].apply(datetime.strptime, args=('%Y-%m-%dT%H:%M:%S.%f',))
    df['time_delta_s'] = (df['timestamp'] - df['timestamp'].shift(1)).apply(lambda x: x.total_seconds())
    df['time_spent'] = df['timestamp'] - df['timestamp'][0]
    df['time_spent_s'] = df['time_spent'].apply(lambda x: x.total_seconds())
    total_duration = df['time_spent_s'].iloc[-1]
    if np.isclose(total_duration, 0):
        df['time_spent_rel'] = 0.
    else:
        df['time_spent_rel'] = df['time_spent_s']/df['time_spent_s'].iloc[-1]

    if 'max_vms' in df.columns:
        df['max_vms_GB'] = df['max_vms'].apply(conversion)
    if 'max_rss' in df.columns:
        df['max_rss_GB'] = df['max_rss'].apply(conversion)
    if 'max_uss' in df.columns:
        df['max_uss_GB'] = df['max_uss'].apply(conversion)
    if 'total_io_read' in df.columns:
        df['total_io_read_GB'] = df['total_io_read'].apply(conversion)
    if 'total_io_write' in df.columns:
        df['total_io_write_GB'] = df['total_io_write'].apply(conversion)
    if 'total_cpu_time' in df.columns:
        df['cpu_perc'] = 100.*(df['total_cpu_time'] - df['total_cpu_time'].shift(1))/df['time_delta_s']

    return df

def get_min_2n(some_number):
    '''  find the minimum power of two greater than a given number '''
    return np.power(2., np.ceil(np.log2(some_number)))


def order_by(attr,stats_list):
	for i in range(1,len(stats_list)):
		for j in range(0, len(stats_list)-i):
			if(stats_list[j].var_dict[attr] < stats_list[j+1].var_dict[attr]):
				k = stats_list[j+1]
				stats_list[j+1] = stats_list[j]
				stats_list[j] = k

	return stats_list

#List of variables to be plotted
VARLIST = ['max_vms_GB', 'max_rss_GB', 'max_uss_GB', 'total_io_read_GB', 'total_io_write_GB',
           'total_cpu_time', 'cpu_perc']

TIME_LIST = ['tottime','cumtime']

ATTR_LIST = ['num_calls','nonrec_calls','tottime','cumtime']

class Stat:
	def __init__(self, key, data):
		self.funct_name = key
		self.var_dict = {}
		for i in range(len(ATTR_LIST)):
			self.var_dict[ATTR_LIST[i]] = data[i]




class UsageParser():

    def __init__(self):
        self.log_files = None
        self.dfs = None
        self.additional_stats = None

        self.time_files = []
        self.group_data = {}
        self.group_names = {}

    def load_log_files(self, wildcard_list, t_fil, max_len=None):
        """ Loads log files that will be used for plotting.

            Parameters
            ----------
            wildcard_list : IDK________________________________________________
                path and name of the files to be used
            max_len :
                Maximum number of log files that will be used.

        """
        time_files = []

        log_files = []


        for wildcard in wildcard_list:
            #log_files += glob.glob(wildcard)
            log_files.append(wildcard)

        for f in t_fil:
            time_files.append(self.load_time_files(f))

        # When maximum length is fixed, get the first max_len files
        if not max_len is None:
            log_files = log_files[:max_len]
            time_files = time_files[:max_len]

        self.log_files = log_files
        """
        dfs = []

        for log_file in t_fil:
            df = pd.read_csv(log_file, engine='python')
            compute_df_columns(df)
            dfs.append(df)

        prueba = dfs
        print ("PRUEBA : ", prueba )
        """

        self.time_files = time_files

        print ("TIME FILES : ", time_files)
        self.load_dfs()

    def load_time_files(self,data_file):
        statsfile = tempfile.NamedTemporaryFile()

        s = open(data_file, 'r+b')

        statsfile.file = s
        statsfile.name = data_file

        rep_list = ['tottime', 'cumtime']

        p2 = pstats.Stats(statsfile.name)
        stats = marshal.load(statsfile.file)
        stats_list = []

        for k in stats:
            stats_list.append(Stat(k, stats[k]))

        ordered_stats = {}

        # se tiene que ordenar para cada atributo que se pasa :
        for var in rep_list:
            ordered_stats[var] = order_by(var, stats_list.copy())

        group_names = {}
        group_data = {}

        for key in ordered_stats:
            for i in range(10):
                if key not in group_names:
                    # print (key, " no existe en la iteracion : ", i, " -- ", ordered_stats['tottime'][0].var_dict['tottime'])
                    group_names[key] = [str(ordered_stats[key][i].funct_name)]
                    group_data[key] = [ordered_stats[key][i].var_dict[key]]
                else:
                    # print (key, " EXISTE en la iteracion : ", i)
                    group_names[key].insert(0, str(ordered_stats[key][i].funct_name))
                    group_data[key].insert(0, ordered_stats[key][i].var_dict[key])


        self.group_data = group_data
        self.group_names = group_names

        return ordered_stats

    def load_dfs(self):
        """ Reads log files and creates dfs list containing every log file
            with data stored in a proper format, ready to be plotted.

            example :
                dfs = [ ] FORMATO !

        """
        dfs = []


        for log_file in self.log_files:
            df = pd.read_csv(log_file, engine='python')
            compute_df_columns(df)
            dfs.append(df)

        self.dfs = dfs

        #print ("LOAD_DFS ", self.dfs, type(self.dfs))



    def plot_sample(self, sample_size=1, var_list=VARLIST, save_plot=False, plot_file=None):

        MARGIN = 0.05

        sample_dfs = random.sample(self.dfs, sample_size)


        print ("SAMPLE DFS : ", sample_dfs)
        n_vars = len(var_list)

        fig = plt.figure(figsize=(8*sample_size, 8*n_vars))

        ax_ind = 1
        for var_name in var_list:
            var_min = np.inf
            var_max = -np.inf
            time_max = -np.inf
            var_axes = []
            for sample_df in sample_dfs:
                ax = fig.add_subplot(n_vars, sample_size, ax_ind)
                var_axes.append(ax)
                ax.plot(sample_df['time_spent_s'], sample_df[var_name])
                ax.set_xlabel('Time Spent')
                ax.set_ylabel(var_name)
                var_max = max([var_max, sample_df[var_name].max()])
                var_min = min([var_min, sample_df[var_name].min()])
                time_max = max([time_max, sample_df['time_spent_s'].max()])
                ax_ind += 1

            var_margin = MARGIN*(var_max - var_min)
            var_lim = [var_min - var_margin, var_max + var_margin]
            time_margin = MARGIN*(time_max)
            time_lim = [-time_margin, time_max + time_margin]
            for ax in var_axes:
                ax.set_ylim(var_lim)
                ax.set_xlim(time_lim)

                if var_name == 'max_uss_GB' :
                    xmin = 0
                    xmax = time_max
                    ymin = ymax = 4
                    line = mlines.Line2D([xmin,xmax], [ymin,ymax], color='green')
                    ax.add_line(line)

        save_or_show(fig, save_plot, plot_file)

    def compute_additional_stats(self, var_list = VARLIST, n_bins=100):

        additional_stats = {}

        # compute mean duration
        durations = [df['time_spent_s'].iloc[-1] for df in self.dfs]
        additional_stats['mean_duration'] = np.mean(durations)
        additional_stats['mean_duration_str'] = timedelta(seconds = additional_stats['mean_duration']).__str__()
        additional_stats['max_duration'] = np.max(durations)
        additional_stats['max_duration_str'] = timedelta(seconds = additional_stats['max_duration']).__str__()

        # compute rss histogram
        if 'max_rss_GB' in var_list:
            mean_rss_hist = np.zeros(n_bins)
            rss_count = 0.
            max_rss_GB = max([df['max_rss_GB'].max() for df in self.dfs])
            print('max_rss_GB: {}'.format(max_rss_GB))
            max_rss_2n = get_min_2n(max_rss_GB)
            hist_bins = np.linspace(0., max_rss_2n, n_bins + 1)
            print('max_rss_2n: {}'.format(max_rss_2n))
            for df in self.dfs:
                h = np.histogram(df['max_rss_GB'], bins=hist_bins)
                rss_hist = h[0].astype(float)
                mean_rss_hist += rss_hist
                rss_count += len(df['max_rss_GB'])
            additional_stats['rss_hist'] = mean_rss_hist/rss_count
            additional_stats['rss_hist_bins'] = hist_bins

            # compute max RSS
            additional_stats['max_rss'] = max([df['max_rss_GB'].max() for df in self.dfs])

        self.additional_stats = additional_stats


    def plot_additional_stats(self, save_plot=False, plot_file=None):

        self.compute_additional_stats()
        hist_bins = self.additional_stats['rss_hist_bins']

        hist_bins_centers = np.array((hist_bins[:-1] + hist_bins[1:])/2)
        fig = plt.figure(figsize=(16., 8.))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(hist_bins_centers, 100*self.additional_stats['rss_hist'], hist_bins_centers[1] - hist_bins_centers[0]);
        ax.set_ylabel('% of sample')
        ax.set_xlabel('GB RSS')

        save_or_show(fig, save_plot, plot_file)

    def plot_value_range(self, var_list=VARLIST, save_plot=False, plot_file=None):

        n_vars = len(var_list)

        fig = plt.figure(figsize=(8, 8*n_vars))

        x = np.linspace(0, 1, 101)
        interp_dfs = {}
        for var_name in var_list:
            interp_dfs[var_name] = pd.DataFrame([], index=x)

        for i_df, df in enumerate(self.dfs):
            for var_name in var_list:
                interp_dfs[var_name][i_df] = np.interp(x, df['time_spent_rel'], df[var_name])

        for i_df, (var_name, interp_df) in enumerate(interp_dfs.items()):
            interp_arr = interp_df.as_matrix()
            ax = fig.add_subplot(n_vars, 1, i_df + 1)
            arr_0 = np.percentile(interp_arr, 0., axis=1)
            arr_25 = np.percentile(interp_arr, 25., axis=1)
            arr_50 = np.percentile(interp_arr, 50., axis=1)
            arr_75 = np.percentile(interp_arr, 75., axis=1)
            arr_100 = np.percentile(interp_arr, 100., axis=1)
            ax.fill_between(x, arr_0, arr_25, color='lightskyblue')
            ax.fill_between(x, arr_25, arr_75, color='steelblue')
            ax.fill_between(x, arr_75, arr_100, color='lightskyblue')
            ax.plot(x, arr_50, color='darkblue')
            ax.set_ylabel(var_name)
            ax.set_xlabel('time %')
            ax.grid(True)
            ax.set_xlim([0., 1.])

        save_or_show(fig, save_plot, plot_file)

    def plot_time(self, sample_size=2, var_list=TIME_LIST):

        ax_ind = 1
        n_vars = len(var_list)
        fig = plt.figure(figsize=(8, 8 * n_vars))

        print (":(")
        for file in self.time_files:
            for key in file :
                ax = fig.add_subplot(n_vars, sample_size, ax_ind)

                #ax.set_xlabel(key)
                ax.legend(self.group_names[key])
                ax.barh(self.group_names[key], self.group_data[key])
                ax_ind += 1
        save_or_show(fig)
