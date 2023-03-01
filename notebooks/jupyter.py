import os
import re
import time
import socket


def get_free_port():
    "finds a free port on the headnode, no root access needed"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    addr = s.getsockname()
    return addr[1]



# submit job and get slurm job ID
jobId = os.popen('sbatch   jupyter.sbatch').read()
host = os.popen('curl http://169.254.169.254/latest/meta-data/public-ipv4').read()
jobId = [int(s) for s in jobId.split() if s.isdigit()][0]
print("jobId = ",jobId)

host = os.popen('curl -s http://169.254.169.254/latest/meta-data/public-ipv4').read()

# wait for the output file to appear
while not os.path.exists(f'jupyter_{jobId}.out'):
    time.sleep(1)

# wait until notebook server is started
content=''
while not "http://127.0.0.1" in content:
   with open(f'jupyter_{jobId}.out') as fh:
      content = fh.read()
   time.sleep(1)

with open(f'jupyter_{jobId}.out') as fh:
   fstring = fh.readlines()

# extract compute node IP
# example: "http://ip-26-0-140-55"
pattern = re.compile(r'(\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3})')
jIP = ''
for line in fstring:
   if pattern.search(line) is not None:
      jIP = pattern.search(line)[0]
      if jIP.startswith('ip-'):
         break
jIP = jIP.split('/ip')[-1].split('/')[0].replace('-','.')
print("job running on cluster compute ip = ",jIP)


# extract jupyter notebook port and token
porttoken = ''
for line in fstring:
   porttoken = line
   if porttoken.startswith(f"     or http://127.0.0.1:"):
      break
porttoken = porttoken.split(':')[-1]
jup_port, token = porttoken.split('/?token=')

hn_port = get_free_port() # lots of users & processes on login node, get a free port
local_port = hn_port      # assume the same port is free on your laptop

username = os.getlogin()

print("\n--- Commands to run: ---")
print ("On your local machine, run this:")
print (f"ssh -L{local_port}:localhost:{hn_port} {username}@{host}")
print()
print ("On the stability login node run:")
print (f"ssh -L{hn_port}:{jIP}:{jup_port} {username}@{jIP}")
print()
print("then browse:")
print (f"http://127.0.0.1:{local_port}/?token={token}")
print()
print("when done, close the job:")
print(f"run: scancel {jobId}")
