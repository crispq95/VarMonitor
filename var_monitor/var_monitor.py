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
        self.var_value = 0.0
        self.clean_report_value()
        self.summary_value = 0.0

    def clean_report_value(self):
        self.report_value = 0.0

    def is_parent(self, some_process):
        if some_process.pid == self.monitor.parent_proc.pid:
            return True
        else:
            return False

    #returns a list with the dead child of a process
    def get_dead_childs(self, some_process):
        if some_process in self.monitor.dead_childs:
            return self.monitor.dead_childs[some_process]

    def reset_dead_childs(self, some_process):
        if some_process in self.monitor.dead_childs:
            self.monitor.dead_childs.pop(some_process)


    '''
    def update_value(self, some_process):
        raise Exception('method not implemented!!')

    def update_summary_value(self):
        raise Exception('method not implemented!!')

    def get_var_value(self):
        raise Exception('method not implemented!!')

    def get_summary_value(self):
        raise Exception('method not implemented!!')
    '''


class RawVarMonitor(VarMonitor):
    def get_var_value(self):
        return self.var_value

    def get_report_value(self):
        return self.report_value

    def get_summary_value(self):
        return self.summary_value


class MemoryVarMonitor(VarMonitor):
    def get_var_value(self):
        return convert_size(self.var_value)

    def get_report_value(self):
        return convert_size(self.report_value)

    def get_summary_value(self):
        return convert_size(self.summary_value)


class MaxRSSMonitor(MemoryVarMonitor):
    def update_value(self, some_process):
        if self.is_parent(some_process):
            self.var_value = some_process.memory_info().rss
        else:
            self.var_value += some_process.memory_info().rss

    def update_report_value(self):
        self.report_value = max(self.var_value, self.report_value)

    def update_summary_value(self):
        self.summary_value = max(self.var_value, self.summary_value)


class MaxVMSMonitor(MaxRSSMonitor):
    def update_value(self, some_process):
        if self.is_parent(some_process):
            self.var_value = some_process.memory_info().vms
        else:
            self.var_value += some_process.memory_info().vms

class MaxUSSMonitor(MaxRSSMonitor):
    def update_value(self, some_process):
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
        self.resta = 0

    def get_process_value(self, some_process):
        raise Exception('Base class does not have this method implemented')

    def set_value_from_value_dict(self):
        # As we have accumulated data for each process
        # it's reasonable to assume that the default aggregator is the sum
        self.var_value = sum(self.var_value_dict.values())

        #for key,val in self.var_value_dict.items():
        #    print("KEY : ", key, "value : ", val)
        #self.var_value_dict = {}


    def update_value(self, some_process):
        d_childs = self.get_dead_childs(some_process)

        cur_val = self.get_process_value(some_process)
        cur_pid = some_process.pid
        resta = 0


        if d_childs:
            for c in d_childs:
                if c.pid in self.var_value_dict:
                    resta += self.var_value_dict.pop(c.pid)
            print("from ", some_process.pid ," -- dead childs ",d_childs," value : ",resta)
            self.reset_dead_childs(some_process)

        #
        #
        # if cur_pid in self.var_value_dict and cur_val < self.var_value_dict[cur_pid]:
        self.var_value_dict[cur_pid] = cur_val - resta
        
        #print ("(",self.resta,")  ANTES : ", self.var_value)
        #print ("(",resta, ")  ANTES : ", self.var_value)
        self.set_value_from_value_dict()
        #print ("DESPUES : ", self.var_value)

    '''
    def update_value(self, some_process):
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
    '''
    def update_report_value(self):
        self.report_value = self.var_value

    def update_summary_value(self):
        self.summary_value = self.var_value


class TotalIOReadMonitor(CumulativeVarMonitor, MemoryVarMonitor):
    def get_process_value(self, some_process):
        return some_process.io_counters().read_chars


class TotalIOWriteMonitor(CumulativeVarMonitor, MemoryVarMonitor):
    def get_process_value(self, some_process):
        return some_process.io_counters().write_chars

    def update_value(self, some_process):
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


class TotalMemSwapMonitor(CumulativeVarMonitor, MemoryVarMonitor):
    def get_process_value(self, some_process):
        return some_process.memory_full_info().swap


class TotalCpuTimeMonitor(CumulativeVarMonitor, RawVarMonitor):
    def get_process_value(self, some_process):
        cpu_times = some_process.cpu_times()
        return cpu_times.user + cpu_times.system

class IOwait(CumulativeVarMonitor, RawVarMonitor):
    def get_process_value(self, some_process):
        return some_process.iowait()


class TotalHS06Monitor(CumulativeVarMonitor, RawVarMonitor):
    def __init__(self, name, proc_monitor):
        print (" ")
        super(TotalHS06Monitor, self).__init__(name, proc_monitor)
        print ("HS06 -- init : name -- ", name, " proc_monitor -- ", proc_monitor)
        print (" ")

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
                                ('total_mem_swap', TotalMemSwapMonitor),
                                ('max_uss', MaxUSSMonitor)])


class ProcessTreeMonitor():

    def __init__(self, proc, var_list, **kwargs):
        #print ("Init_ : ", proc, var_list, kwargs)
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
        child_list = []
        temp_dead_childs = []
        self.dead_childs = {}
        old_process_tree = self.process_tree.copy()
        self.process_tree = {}

        for c in self.parent_proc.children():
            if c.is_running() : #si el hijo esta vivo
                child_list.append(c)
            else :
                temp_dead_childs.append(c)

        if child_list :
            self.process_tree[self.parent_proc] = child_list
        if temp_dead_childs:
            self.dead_childs[self.parent_proc] = temp_dead_childs

        l_act = {}
        l_act = self.process_tree.copy()

        while (l_act):
            nodes = l_act.popitem()
            for n in nodes[1]:
                child_list = []
                if n.is_running():
                    if n.children():
                        for child in n.children():
                            if child.is_running():
                                child_list.append(child)
                        if child_list:
                            l_act[n] = child_list
                            self.process_tree[n] = child_list
                    else:
                        self.process_tree[n] = []

        for parent,children in old_process_tree.items():
            temp_dead_childs = []

            for child in children :
                if not child.is_running():
                    temp_dead_childs.append(child)

            self.dead_childs[parent] = temp_dead_childs

        '''
        print ("OLD PTREE : ", old_process_tree)
        print (" ")
        print ("PROCESS TREE -- ", self.process_tree)
        print ("Dead CHIDLS : ", self.dead_childs)
        print (" ")

        
      
        for key,childs in  aux_dic.items():
            child_list = []
            temp_dead_childs = []

            for child in childs:
                if child.is_running():
                    if child.children():
                        for c in child.children() :
                            if child.is_running():
                                child_list.append(c)
                            else :
                                temp_dead_childs.append(c)

                        if child_list:
                            new_process_tree[child] = child_list
                        if temp_dead_childs:
                            self.dead_childs[child] = temp_dead_childs
                else :
                    if key in self.dead_childs:
                        self.dead_childs[key].append(child)
                    else :
                        self.dead_childs[key] = [child]
        '''
        #self.process_tree = new_process_tree
        #print ("PROCESS TREE : ", self.process_tree)
        #print ("DEAD CHILDS : ", dead_childs)

    def update_values(self, some_process):
        for monitor in self.monitor_list:
            monitor.update_value(some_process)

    def update_report_values(self):
        for monitor in self.monitor_list:
            monitor.update_report_value()

    def update_summary_values(self):
        for monitor in self.monitor_list:
            monitor.update_summary_value()

    def clean_report_values(self):
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
        self._log_file.write(self.get_headers())
        time_report = datetime.datetime.now()

        self.init_process_tree()

        while self.proc_is_running():
            self.update_process_tree()

            '''
            print ("PROCESS TREE : ", self.process_tree)
            if self.dead_childs:
                print ("OUTSIDE D_CHILDS : ", self.dead_childs)
                print (" ")
            '''

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

                '''
                print ("PROCESS TREE : ", self.process_tree)
                print (" ")
                print ("DEAD CHILDS : ", self.dead_childs)
                print (" ")
                '''
            time.sleep(self.check_lapse)

        self.parent_proc.wait()
        print (" ")

    def proc_is_running(self):
        return self.parent_proc.is_running() and self.parent_proc.status() != psutil.STATUS_ZOMBIE


