#!/usr/bin/python

import os
import psutil
import time
import math
import datetime
import logging
from collections import OrderedDict
import shlex
import subprocess as sp
import re
import sys
import threading

CHECK_LAPSE = 0  # time between each usage check in seconds
REPORT_LAPSE = 1  # time between each usage print in seconds


def convert_size(size_bytes):
    """Converts bytes into the most appropriate format.

        Parameters
        ----------
        size_bytes : float
            Number of Bytes to be converted.

        Returns
        -------
        return : str
            String with the converted data represented as a number with a maximum of
            two decimals and a letter showing the chosen size (B = bytes, K = kilobytes, etc.)

            example :
                size_bytes = 6836224
                return = 6.52M
    """

    if (size_bytes == 0):
        return '0B'
    size_name = ("B", "K", "M", "G", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return '%s%s' % (s, size_name[i])


class VarMonitor(object):
    def __init__(self, name, proc_monitor):
        self.name = name
        self.reset_values()
        self.monitor = proc_monitor

    def reset_values(self):
        """ Sets monitor values to 0 """
        self.var_value = 0.0
        self.clean_report_value()
        self.summary_value = 0.0

    def clean_report_value(self):
        """ Sets report value to 0 """
        self.report_value = 0.0

    def is_parent(self, some_process):
        """Checks if a given process is the parent of all the other running processes.

        Parameters
        ----------
        some_process : psutil.Process

        Returns
        -------
        return : bool
                    - True : some_process is the 1st parent process
                    - False : some_process is not the 1st the parent process
        """
        if some_process.pid == self.monitor.parent_proc.pid:
            return True
        else:
            return False

    def get_dead_childs(self, some_process):
        """Returns a list with the dead children of a given process"""
        if some_process in self.monitor.dead_childs:
            return self.monitor.dead_childs[some_process]

    def reset_dead_childs(self, some_process):
        """  Removes the child dead children from a given process of the dead_childs dict """
        if some_process in self.monitor.dead_childs:
            self.monitor.dead_childs.pop(some_process)


class RawVarMonitor(VarMonitor):
    def get_var_value(self):
        """ Returns var_value in Bytes"""
        return self.var_value

    def get_report_value(self):
        """ Returns report_value in Bytes"""
        return self.report_value

    def get_summary_value(self):
        """ Returns summary_value in Bytes"""
        return self.summary_value


class MemoryVarMonitor(VarMonitor):
    def get_var_value(self):
        """ Returns var_value in a convenient format to be printed or stored"""
        return convert_size(self.var_value)

    def get_report_value(self):
        """ Returns report_value in a convenient format to be printed or stored"""
        return convert_size(self.report_value)

    def get_summary_value(self):
        """ Returns summary_value in a convenient format to be printed or stored"""
        return convert_size(self.summary_value)


class MaxRSSMonitor(MemoryVarMonitor):
    def update_value(self, some_process):
        """ Stores on var_value the current value of the non-swapped
            physical memory used by a given process """
        if self.is_parent(some_process):
            self.var_value = some_process.memory_info().rss
        else:
            self.var_value += some_process.memory_info().rss

    def update_report_value(self):
        """ Stores on report_value the maximum between report_value and var_value """
        self.report_value = max(self.var_value, self.report_value)

    def update_summary_value(self):
        """ Stores on summary_value the maximum between itself and var_value """
        self.summary_value = max(self.var_value, self.summary_value)


class MaxVMSMonitor(MaxRSSMonitor):
    def update_value(self, some_process):
        """ Stores on var_value the current value of the virtual
            memory used by a given process """
        if self.is_parent(some_process):
            self.var_value = some_process.memory_info().vms
        else:
            self.var_value += some_process.memory_info().vms


#MOST REPRESENTATIVE PARAMETER FOR DETERMINING HOW MANY MEMORY IS ACTUALLY USED BY A PROCESS
class MaxUSSMonitor(MaxRSSMonitor):
    def update_value(self, some_process):
        """ Stores on var_value the current value of memory wich is unique to the given process
         and would be freed if the process was terminated right now """
        if self.is_parent(some_process):
            self.var_value = some_process.memory_full_info().uss
        else:
            self.var_value += some_process.memory_full_info().uss

class CumulativeVarMonitor(VarMonitor):
    def reset_values(self):
        self.var_value = 0.0
        self.var_value_dict = {}
        self.report_value = 0.0
        self.summary_value = 0.0
        self.backup_count = 0

    def get_process_value(self, some_process):
        raise Exception('Base class does not have this method implemented')

    def set_value_from_value_dict(self):
        "Updates var_value as the sum of all the values from var_value_dict"
        # As we have accumulated data for each process
        # it's reasonable to assume that the default aggregator is the sum
        self.var_value = sum(self.var_value_dict.values())

    def update_value(self, some_process):
        """ Updates var_value_dict for a given process acording to its current value.
            Calls set_value_from_dict() to update var_value of the given process

            Parameters
            ----------
            some_process : psutil.Process
               Process to be monitorized
        """
        cur_val = self.get_process_value(some_process)
        cur_pid = some_process.pid

        if cur_pid in self.var_value_dict and cur_val < self.var_value_dict[cur_pid]:
            # if the current value is lower than the already existent, it means
            # that the pid has been reused
            # move the old value to a backup
            bk_pid = '{}_{}'.format(cur_pid, self.backup_count)
            self.var_value_dict[bk_pid] = self.var_value_dict[cur_pid]
            self.backup_count += 1

        self.var_value_dict[cur_pid] = cur_val
        self.set_value_from_value_dict()

    def update_report_value(self):
        """ Updates report_value with the current var_value """
        self.report_value = self.var_value

    def update_summary_value(self):
        """ Updates summary_value with the current var_value """
        self.summary_value = self.var_value


class IOCumulativeVarMonitor(CumulativeVarMonitor,VarMonitor):
    def reset_values(self):
        self.var_value = 0.0
        self.var_value_dict = {}
        self.report_value = 0.0
        self.summary_value = 0.0
        self.backup_count = 0

    def update_value(self, some_process):
        """ Calculates the current value for IO monitors and updates var_value.

            This function considers that IO monitors are cummulative (they accumulate
            their children values once childs are dead) and calculates the IO value for
            a given process.

            Parameters
            ----------
            some_process : psutil.Process
                Process to be monitorized

        """

        #dead childs list
        d_childs = self.get_dead_childs(some_process)

        #current value of IO for the given process
        cur_val = self.get_process_value(some_process)
        cur_pid = some_process.pid

        #current value of the data read/written by the given process's childs
        resta = 0

        #for all the dead childs belonging to a certain process accummulates their IO value
        if d_childs:
            for c in d_childs:
                if c.pid in self.var_value_dict:
                    if c.pid in self.var_value_dict:
                        resta += self.var_value_dict.pop(c.pid)

        #Updates var_value_dict for a given process considering the IO value for its dead childs
        self.var_value_dict[cur_pid] = cur_val - resta

        #updates var_value
        self.set_value_from_value_dict()



class TotalIOReadMonitor(IOCumulativeVarMonitor, MemoryVarMonitor):
    def get_process_value(self, some_process):
        """ Returns the number of read bytes of a given process """
        return some_process.io_counters().read_chars


class TotalIOWriteMonitor(IOCumulativeVarMonitor, MemoryVarMonitor):
    def get_process_value(self, some_process):
        """ Returns the number of written bytes of a given process """
        return some_process.io_counters().write_chars


class TotalCpuTimeMonitor(CumulativeVarMonitor, RawVarMonitor):
    def get_process_value(self, some_process):
        """" Returns the sum of the cpu time used by the given process executing on kernel and user modes """
        cpu_times = some_process.cpu_times()
        return cpu_times.user + cpu_times.system



class TotalHS06Monitor(CumulativeVarMonitor, RawVarMonitor):
    def __init__(self, name, proc_monitor):
        super(TotalHS06Monitor, self).__init__(name, proc_monitor)

        # Get HS06 factor
        # get the script to find the HS06 factor and run it
        HS06_factor_command_list = shlex.split(proc_monitor.kwargs.get('HS06_factor_func'))

        print (HS06_factor_command_list, sp.PIPE)

        p = sp.Popen(HS06_factor_command_list, stdout=sp.PIPE, stderr=sp.PIPE)

        print (p)
        p.wait()

        # Capture the HS06 factor from the stdout
        m = re.search('HS06_factor=(.*)', p.stdout.read())
        self.HS06_factor = float(m.group(1))

    def get_process_value(self, some_process):
        # get CPU time
        cpu_times = some_process.cpu_times()

        # compute HS06*h
        return self.HS06_factor * (cpu_times.user + cpu_times.system) / 3600.0

    def get_var_value(self):
        return '{:.4f}'.format(self.var_value)

    def get_summary_value(self):
        return '{:.4f}'.format(self.summary_value)


VAR_MONITOR_DICT = OrderedDict([('max_vms', MaxVMSMonitor),
                                ('max_rss', MaxRSSMonitor),
                                ('total_io_read', TotalIOReadMonitor),
                                ('total_io_write', TotalIOWriteMonitor),
                                ('total_cpu_time', TotalCpuTimeMonitor),
                                ('total_HS06', TotalHS06Monitor),
                                ('max_uss', MaxUSSMonitor)])


class ProcessTreeMonitor():

    def __init__(self, proc, var_list, **kwargs):


        self.parent_proc = proc
        self.kwargs = kwargs
        self.monitor_list = [VAR_MONITOR_DICT[var](var, self) for var in var_list]
        self.report_lapse = kwargs.get('report_lapse', REPORT_LAPSE)
        self.check_lapse = kwargs.get('check_lapse', CHECK_LAPSE)
        if 'log_file' in kwargs:
            if os.path.exists(kwargs['log_file']):
                raise Exception('File {} already exists'.format(kwargs['log_file']))
            self._log_file = open(kwargs['log_file'], 'a+')
        else:
            self._log_file = sys.stdout
        self.lock = threading.RLock()

        self.process_tree = {}
        self.dead_childs = {}

    def init_process_tree(self):
        """Creates an initial process tree to keep track of all the processes we need to monitorize

            Process tree will be stored on self.process_tree
        """

        child_list = []

        for c in self.parent_proc.children():
            if c.is_running() : #si el hijo esta vivo
                child_list.append(c)

        if child_list :
            self.process_tree[self.parent_proc] = child_list
        aux_dic = self.process_tree.copy()

        for key,childs in  aux_dic.items():
            child_list = []
            for child in childs:
                if child.is_running():
                    if child.children():
                        for c in child.children() :
                            if child.is_running():
                                child_list.append(c)
                        if child_list:
                            self.process_tree[child] = child_list

    def update_process_tree(self):
        """ Updates the process tree and the dictionary of dead childs of the monitorized program.

            Used to keep track of the processes running on the monitorized program and their hierarchy.
            dead_childs dictionary will be used for a real-time monitorization of the IO of the program.


            example :

                self.process_tree = { parent:[child1,child2,...,childN], parent2:[child1,...,childN], ... }
                self.dead_childs got the same format shown above but with dead childs instead of the alive ones

        """

        child_list = []         #temporary list for alive childs of a given process
        temp_dead_childs = []   #temporary list for dead childs of a given process

        self.dead_childs = {}
        #used to keep track of the processes that are already dead
        old_process_tree = self.process_tree.copy()

        self.process_tree = {}


        #inits the process_tree dictionary with the parent process of the program childrens'
        for c in self.parent_proc.children():
            if c.is_running() : #si el hijo esta vivo
                child_list.append(c)
            else :
                temp_dead_childs.append(c)

        if child_list :
            self.process_tree[self.parent_proc] = child_list
        if temp_dead_childs:
            self.dead_childs[self.parent_proc] = temp_dead_childs


        #l_act = parent : [child1, child2, ... , childN]
        l_act = {}
        l_act = self.process_tree.copy()


        # Performs a search for all the alive childs of each node in the process tree and add
        # them on self.process_tree
        while (l_act):  #while there are parents to check
            nodes = l_act.popitem() #pop a parent and their childs

            for n in nodes[1]:  # n = each child of the popped parent
                child_list = []

                # if the child is running and has childs
                if n.is_running():
                    if n.children():
                        for child in n.children():
                            if child.is_running():
                                # check for every child if its alive and add it to the child list
                                child_list.append(child)
                        #if we've found alive childs
                        if child_list:
                            #add them with their parent to the list of actual nodes to be checked
                            l_act[n] = child_list
                            #add them to the actual process_tree
                            self.process_tree[n] = child_list
                    else:
                        self.process_tree[n] = []


        # checks for dead children comparing the old_process_tree vs the one found above
        # to update the dead childs dictionary (self.dead_childs)
        for parent,children in old_process_tree.items():
            temp_dead_childs = []

            for child in children :
                if not child.is_running():
                    temp_dead_childs.append(child)

            self.dead_childs[parent] = temp_dead_childs

    def update_values(self, some_process):
        " Calls update_value for a given process for all the variables to monitorize "
        for monitor in self.monitor_list:
            monitor.update_value(some_process)

    def update_report_values(self):
        " Calls update_report_value for all the variables to monitorize "
        for monitor in self.monitor_list:
            monitor.update_report_value()

    def update_summary_values(self):
        " Calls update_summary_value for all the variables to monitorize "
        for monitor in self.monitor_list:
            monitor.update_summary_value()

    def clean_report_values(self):
        " Calls clean_report_value for all the variables to monitorize "
        for monitor in self.monitor_list:
            monitor.clean_report_value()

    def get_var_values(self):
        return ', '.join(['{}, {}'.format(monit.name, monit.get_var_value()) for monit in self.monitor_list])

    def get_report_values(self):
        s = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f') + ','
        return s + ','.join(['{}'.format(monit.get_report_value()) for monit in self.monitor_list]) + '\n'

    def get_summary_values(self):
        return ', '.join(['{}, {}'.format(monit.name, monit.get_summary_value()) for monit in self.monitor_list])

    def get_headers(self):
        return 'timestamp,' + ','.join([monit.name for monit in self.monitor_list]) + '\n'

    def update_all_values(self):
        """ Updates the values for all the processes running on the monitorized program.

            Calls update_values for the parent process of the program and all its children
            if a children is no longer alive its ignored and keeps calling the next one.

            Once all processes have updated their monitorized values this function also
            calls on updating for report and summary values.

        """
        # get var values from parent process
        self.update_values(self.parent_proc)

        # iterate over children and update their values
        children_process_list = self.parent_proc.children(recursive=True)
        for children_process in children_process_list:
            try:
                self.update_values(children_process)
            except:
                pass

        # update report values
        self.update_report_values()

        # update summary values
        self.update_summary_values()

    def write_log(self, log_message):
        self.lock.acquire()
        try:
            self._log_file.write(log_message)
            if hasattr(self._log_file, 'flush'):
                self._log_file.flush()
        finally:
            self.lock.release()

    def start(self):
        """ Main function for ProcessTreeMonitor() class

            Monitorizes each chosen parameter by calling on update_all_values() and
            all the funcions needed to work correctly with a frequency determined
            by CHECK_LAPSE. It updates all the report and summary values while the
            parent process is still alive and running.
        """

        self._log_file.write(self.get_headers())
        time_report = datetime.datetime.now()

        self.init_process_tree()

        #while the parent process is alive and running
        while self.proc_is_running():
            # update the process tree of the program
            self.update_process_tree()


            # calls update_all_values() to update current values and report/summary values on
            # all processes running on the program
            try:
                self.update_all_values()
            except psutil.AccessDenied:
                pass

            # print usage if needed
            now = datetime.datetime.now()
            if (now - time_report).total_seconds() > self.report_lapse:
                self.write_log(self.get_report_values())
                self.clean_report_values()
                time_report = now
            # sleeps until is the time for the next usage check
            time.sleep(self.check_lapse)

        #waits until all the child processes are terminated
        self.parent_proc.wait()
        print (" ")

    def proc_is_running(self):
        """ Checks if the parent process of the program is still running and it is still alive """
        return self.parent_proc.is_running() and self.parent_proc.status() != psutil.STATUS_ZOMBIE


