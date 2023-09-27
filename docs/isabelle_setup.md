# Special Setup: Isabelle Proof Checker 


**A special setup is required for evaluating the `miniF2F_isabelle_informal2formal` task.**


Below, we assume that you have run the evaluation harness on the `miniF2F_isabelle_informal2formal` task.

The task produces an output JSON file containing the generated proofs, e.g. `output/minif2f_isabelle/codellama_CodeLlama-7b-hf.json`).

We will now set up proof checking in order to run `unsafe_score_minif2f_isabelle.py`, which evaluates the proofs. 

## Setup

The `unsafe_score_minif2f_isabelle.py` script supports proof checking via [PISA](https://github.com/albertqjiang/Portal-to-ISAbelle/tree/56def2c39f85d211e1f40cc5765581a567879106). We implement a client that interacts with PISA (`Checker` in script).

Here are setup steps for a non-dockerized environment. The setup is heavily based on the [PISA readme](https://github.com/albertqjiang/Portal-to-ISAbelle/tree/56def2c39f85d211e1f40cc5765581a567879106)  and [Dockerfile](https://github.com/albertqjiang/Portal-to-ISAbelle/blob/main/docker/Dockerfile). You may need to refer to those if something goes wrong.

### Installation (PISA and Isabelle)
First, we need to set up PISA and Isabelle.
```bash
# -- PISA setup
# Download Portal-to-ISAbelle (PISA)
# This version includes an extended timeout:
cd ~/
git clone https://github.com/wellecks/Portal-to-ISAbelle
cd Portal-to-ISAbelle
git checkout aa5a06d217af4b04f4ae573836b52572da0858f6

# Scala installation
sudo apt-get install zip
curl -s "https://get.sdkman.io" | bash
source "~/.sdkman/bin/sdkman-init.sh"
sdk install java 11.0.11-open
sdk install sbt

# Compile PISA 
cd ~/Portal-to-ISAbelle
sbt compile
sbt assembly

# -- Isabelle setup
# Download Isabelle
wget https://isabelle.in.tum.de/website-Isabelle2022/dist/Isabelle2022_linux.tar.gz && \
    tar -xzf Isabelle2022_linux.tar.gz

# Install Isabelle (i.e., move to WORK_DIR, make an alias).
export WORK_DIR=~/
mv Isabelle2022 ${WORK_DIR}/
echo 'alias isabelle=${WORK_DIR}/Isabelle2022/bin/isabelle' >> ~/.bashrc
source ~/.bashrc

# Build Isabelle HOL (creates heaps in ~/.isabelle)
isabelle build -b -D ${WORK_DIR}/Isabelle2022/src/HOL/ -j 20
```

At the end, here's what the setup looks like:
- Portal-to-ISAbelle github repo in `~/Portal-to-ISAbelle`
- Isabelle in `~/Isabelle2022`, e.g.
    ```
    ls ~/Isabelle2022
      
    => ANNOUNCE  bin  contrib ...
    ```
- Isabelle heaps in `~/.isabelle`, e.g.
    ```
    ls ~/.isabelle/Isabelle2022/heaps/polyml-5.9_x86_64_32-linux/
  
    => Group-Ring-Module  HOL-Corec_Examples  HOL-Isar_Examples  ...
    ```
  

### Start a PISA server
Now start a PISA server:
```bash
cd ~/Portal-to-ISAbelle
sbt "runMain pisa.server.PisaOneStageServer9000"
```
The number at the end (here, 9000) specifies the server's port.

Next, start a separate tmux window. In this window, we will configure the Python client to communicate with the server, then run the `unsafe_score_minif2f_isabelle.py` script.
### Configuration


At a high-level, we have three components:
1. The PISA Scala server
2. The PISA python library 
3. Our python client, Checker.


#### Set PISA_PATH

First, set a `PISA_PATH` environment variable that points to PISA's python directory:
```bash
export PISA_PATH=~/Portal-to-ISAbelle/src/main/python
```
The variable is used to import PISA's python client (`Portal-to-Isabelle/src/main/python/pisa_client.py`) in Checker. \
This links components 2 and 3.


#### Setup a working directory and working file
PISA is initialized by providing a particular working directory and file. \
We will create a file called `Interactive.thy` and put it in the `HOL/Examples` directory:

```bash
vim ~/Isabelle2022/src/HOL/Examples/Interactive.thy
```
```
theory Interactive
  imports Complex_Main
begin

end
```

### Run the evaluation:

#### Install misc. libraries
```bash
pip install func_timeout
```

#### Run the script:
As command line arguments we pass the path to Isabelle, the working directory described above, the theory file described above, the port of the running PISA server, and our output JSON file:
```bash
python unsafe_score_minif2f_isabelle.py \
  --isa-path /home/username/Isabelle2022 \
  --theory-file /home/username/Isabelle2022/src/HOL/Examples/Interactive.thy \
  --working-dir /home/username/Isabelle2022/src/HOL/Examples \
  --port 9000 \
  --output output/minif2f_isabelle/codellama_CodeLlama-7b-hf.json
```
Naturally, you will need to set these arguments to your own file paths.

In some environments it may be necessary to prefix the command with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
