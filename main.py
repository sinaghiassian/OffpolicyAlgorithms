import os

from Job.JobBuilder import JobBuilder

if __name__ == "__main__":
    builder = JobBuilder(json_path=os.getcwd() + '\\Experiments\\FirstChain\\TD\\config.json')
    print(builder())