import subprocess
import json
import os
import datetime
import time
from bench_util import *
import os
import exp_config

from variables import *


class BaseRun:
    def __init__(self, exp_id) -> None:
        self.exp_id = exp_id

    def run(self,):
        raise NotImplementedError


class Experiment:
    registered_id = []

    def __init__(self, exp_id) -> None:
        if exp_id not in self.registered_id:
            self.registered_id.append(exp_id)
        else:
            # raise Exception("repeated exp ID")
            pass

        self.exp_id = exp_id
        self.log_dir = None
        pass

    def _BeforeStartAllRun(self, ) -> None:
        raise NotImplementedError

    def _SortConfigs(self, runs):
        raise NotImplementedError

    def _RunHook(self, previous_run, next_run):
        raise NotImplementedError

    def _AllRuns(self,):
        raise NotImplementedError

    def SetLogDir(self, log_dir):
        self.log_dir = log_dir

    def LenAllRuns(self):
        return len(self._AllRuns())

    def RunExperiment(self):
        if self.log_dir is None:
            raise Exception("invalid log dir")
        os.makedirs(self.log_dir, exist_ok=True)
        self._BeforeStartAllRun()
        all_runs = self._AllRuns()
        print("len of all configs = ", len(all_runs))
        previous_run = None
        for each_i, each_run in enumerate(all_runs):
            self._RunHook(previous_run, each_run)
            print(datetime.datetime.now(),
                  f'EXP{self.exp_id}: {each_i} / {len(all_runs)}', flush=True)

            ret = os.path.isfile("/dev/shm/fyy_is_using")
            if ret:
                print("wait fyy finish")
            while ret:
                ret = os.path.isfile("/dev/shm/fyy_is_using")
                time.sleep(10)
            print("wait fyy finish escape")

            each_run.run()
            previous_run = each_run

        self._RunHook(previous_run, None)


class CSRun(BaseRun):
    def __init__(self, ps_servers, client_servers,
                 exp_id, run_id, log_dir,
                 server_config, client_config,
                 server_bin_path, client_bin_path
                 ) -> None:
        super().__init__(exp_id)
        self.run_id = run_id
        self.log_dir = log_dir
        self.server_config = server_config
        self.client_config = client_config
        self.ps_servers = ps_servers
        self.client_servers = client_servers
        self.server_bin_path = server_bin_path
        self.client_bin_path = client_bin_path
        self.check_config()

    def check_config(self,):
        if len(self.ps_servers) == 0 or len(self.client_servers) == 0:
            raise Exception("no machines")

        # wait a condition to start client processes
    def _ClientWaitServer(self):
        raise NotImplementedError

    def run(self, ):
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"mkdir {self.log_dir}")
        time.sleep(1)
        # dump config
        dumped_config = {
            "server":
                self.server_config,
            "client": self.client_config,
        }
        dir_path = os.path.dirname(os.path.realpath(__file__))

        subprocess.run(
            f"bash {dir_path}/../third_party/Mayfly-main/script/restartMemc.sh", shell=True, check=True)

        global_id = 0
        for ps_id, (each_host, numa_id) in enumerate(self.ps_servers):
            config = ' '.join(
                [f'--{k}={v}' for k, v in self.server_config.items()])
            server_command = f'''{self.server_bin_path} --numa_id={numa_id} --global_id={global_id} \
            --num_server_processes={len(self.ps_servers)} --num_client_processes={len(self.client_servers)} \
            {config} >{self.log_dir}/ps_{ps_id} 2>&1'''
            RemoteExecute(each_host, server_command, PROJECT_PATH)
            dumped_config[f'server_{ps_id}'] = server_command
            global_id += 1

        self._ClientWaitServer()

        for client_id, (each_host, numa_id) in enumerate(self.client_servers):
            config = ' '.join(
                [f'--{k}={v}' for k, v in self.client_config.items()])
            client_command = f'''{self.client_bin_path} --numa_id={numa_id} --global_id={global_id} \
            --num_server_processes={len(self.ps_servers)} --num_client_processes={len(self.client_servers)} \
             {config} >{self.log_dir}/client_{client_id} 2>&1 &'''
            RemoteExecute(each_host, client_command, PROJECT_PATH)
            dumped_config[f'client_{ps_id}'] = client_command
            global_id += 1

        with open(f'{self.log_dir}/config', 'w') as f:
            import json
            json.dump(dumped_config, f, indent=2)


class CSExperiment(Experiment):
    def __init__(self, exp_id, common_config, server_config, client_config, ps_servers, client_servers) -> None:
        super().__init__(exp_id)
        self.common_config = common_config
        self.server_config = server_config
        self.client_config = client_config
        self.ps_servers = ps_servers
        self.client_servers = client_servers

    def _AllRuns(self,):
        return list(self.get_next_config())

    def _PostprocessConfig(self, server_configs, client_configs):
        # don't use self
        raise NotImplementedError

    def _CreateRun(self, run_id, run_log_dir, run_server_config, run_client_config,):
        raise NotImplementedError

    def get_next_config(self, ):
        common_config = PreprocessConfig(self.common_config)
        server_config = PreprocessConfig(self.server_config)
        client_config = PreprocessConfig(self.client_config)

        # [(dictA, dictB), (dictA, dictB), (dictA, dictB),]
        print("server_config has ", len(server_config), "configs")
        print("client_config has ", len(client_config), "configs")

        def find_start_run_id():
            import re
            ids = [int(re.search(r'(\d+)', each)[1])
                   for each in os.listdir(self.log_dir)]
            if len(ids) != 0:
                max_id = max(ids)
            if len(ids) == 0:
                max_id = -1
            print(os.listdir(self.log_dir))
            return max_id + 1

        run_id = find_start_run_id()
        print(f"-------start run_id ={run_id} ====================")
        runs = []
        for each_common_config in common_config:
            for each_server_config in server_config:
                for each_client_config in client_config:
                    each_server_config_copy = each_server_config.copy()
                    each_client_config_copy = each_client_config.copy()
                    each_server_config_copy.update(each_common_config)
                    each_client_config_copy.update(each_common_config)
                    # add custom process
                    self._PostprocessConfig(
                        each_server_config_copy, each_client_config_copy)

                    runs.append(self._CreateRun(run_id, os.path.join(
                        self.log_dir, f'run_{run_id}'), each_server_config_copy, each_client_config_copy))
                    run_id += 1
        for each in runs:
            yield each


class LocalOnlyRun(BaseRun):
    def __init__(self,
                 exp_id, run_id, log_dir,
                 config,
                 bin_path, pwd_path, execute_host
                 ) -> None:
        super().__init__(exp_id)
        self.run_id = run_id
        self.log_dir = log_dir
        self.config = config
        self.bin_path = bin_path
        self.execute_host = execute_host
        self.pwd_path = pwd_path
        self.check_config()

    def check_config(self,):
        pass

    def run(self, ):
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"mkdir {self.log_dir}")
        time.sleep(1)
        # dump config
        dumped_config = self.config

        config = ' '.join(
            [f'--{k}={v}' for k, v in self.config.items()])
        server_command = f'''{self.bin_path} \
        {config} >{self.log_dir}/log 2>&1 &'''

        dumped_config['command'] = server_command
        LocalExecute(server_command, self.pwd_path)

        with open(f'{self.log_dir}/config', 'w') as f:
            import json
            json.dump(dumped_config, f, indent=2)


class LocalOnlyExperiment(Experiment):
    def __init__(self, exp_id, common_config, execute_host) -> None:
        super().__init__(exp_id)
        self.common_config = common_config
        self.execute_host = execute_host

    def _AllRuns(self,):
        return list(self.get_next_config())

    def _PostprocessConfig(self, server_configs, client_configs):
        # don't use self
        raise NotImplementedError

    def _CreateRun(self, run_id, run_log_dir, run_config,):
        raise NotImplementedError

    def _FindAllRunedConfig(self, ):
        configs = []
        for each in os.listdir(self.log_dir):
            if each.startswith("run_"):
                with open(os.path.join(self.log_dir, each, "log"), 'r') as f:
                    content = f.read()
                if "Succ" not in content:
                    print(f"no success {os.path.join(self.log_dir, each)}")
                    # LocalExecute(f"rm -rf {os.path.join(self.log_dir, each)}")
                    continue

                with open(os.path.join(self.log_dir, each, "config"), 'r') as f:
                    config = json.load(f)
                    config.pop('command')
                    configs.append(config)
        return configs


    def get_next_config(self, ):
        configs = PreprocessConfig(self.common_config)

        # [(dictA, dictB), (dictA, dictB), (dictA, dictB),]
        print("common_config has ", len(configs), "configs")

        def find_start_run_id():
            import re
            ids = []
            for each in os.listdir(self.log_dir):
                re_r = re.search(r'(\d+)', each)
                if re_r is None:
                    continue
                ids.append(int(re_r[1]))
            if len(ids) != 0:
                max_id = max(ids)
            if len(ids) == 0:
                max_id = -1
            print(os.listdir(self.log_dir))
            return max_id + 1

        self.runned_configs = self._FindAllRunedConfig()

        run_id = find_start_run_id()
        print(f"-------start run_id ={run_id} ====================")
        runs = []
        all_configs = self._SortConfigs(configs)
        for each_config in all_configs:
            # add custom process
            self._PostprocessConfig(each_config, )

            if each_config in self.runned_configs:
                print("already runned, continue next run")
                continue

            runs.append(self._CreateRun(run_id, os.path.join(
                self.log_dir, f'run_{run_id}'), each_config, self.execute_host))
            run_id += 1
        for each in runs:
            yield each
