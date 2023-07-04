# Special Setup: Isabelle Proof Checker 


**A special setup is _required_ for tasks that use Isabelle proof checking:**

- `miniF2F_isabelle`

Follow this guide to set up Isabelle proof checking.

## Setup

The evaluation harness supports proof checking via [PISA](https://github.com/albertqjiang/Portal-to-ISAbelle/tree/56def2c39f85d211e1f40cc5765581a567879106). We implement a client that interacts with PISA (`Checker` in [minif2f_isabelle.py](https://github.com/wellecks/lm-evaluation-harness/blob/minif2f-isabelle/lm_eval/tasks/minif2f_isabelle.py#L154)).

Here are setup steps for a non-dockerized environment. The setup is heavily based on the [PISA readme](https://github.com/albertqjiang/Portal-to-ISAbelle/tree/56def2c39f85d211e1f40cc5765581a567879106)  and [Dockerfile](https://github.com/albertqjiang/Portal-to-ISAbelle/blob/main/docker/Dockerfile). You may need to refer to those if something goes wrong.

### Installation (PISA and Isabelle)
First, we need to set up PISA and Isabelle.
```bash
# -- PISA setup
# Download Portal-to-ISAbelle (PISA)
cd ~/
git clone https://github.com/albertqjiang/Portal-to-ISAbelle.git

# Scala installation
sudo apt-get install zip
curl -s "https://get.sdkman.io" | bash
source "/home/seanw/.sdkman/bin/sdkman-init.sh"
sdk install java 11.0.11-open
sdk install sbt

# Compile PISA 
cd ~/Portal-to-ISAbelle
sbt compile
sbt assembly

# -- Isabelle setup
# Download Isabelle
wget https://isabelle.in.tum.de/dist/Isabelle2022_linux.tar.gz && \
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
You can test out the installation so far by starting a PISA server:
```bash
cd ~/Portal-to-ISAbelle
sbt "runMain pisa.server.PisaOneStageServer9000"
```

The next step is to specify a configuration that allows the Python client to talk to the Scala PISA server, as described below.

### Configuration

At a high-level, we have three components:
1. The PISA Scala server
2. The PISA python library 
3. Our python client, [Checker](https://github.com/wellecks/lm-evaluation-harness/blob/minif2f-isabelle/lm_eval/tasks/minif2f_isabelle.py#L154)

We need to set environment variables and configuration so that all three can talk to each other.

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
We will use this working directory and file in the next step.

#### Setup a `config.json`

Next, we need to specify the path to Isabelle, the working directory, and the working file (theory file). \
These are used to initialize a working Isabelle instance. This links components 1 and 2.

To do so, specify a `isabelle_checker` field in `configs/config/minif2f_isabelle.json`. \
For example (here, the home directory `~` is `/home/seanw`):
```json
{
    "minif2f_isabelle": {
        "description": "...prompt...",
        "params": {
            "isabelle_checker" : {
                "isa_path": "/home/seanw/Isabelle2022",
                "working_dir": "/home/seanw/Isabelle2022/src/HOL/Examples",
                "theory_file": "/home/seanw/Isabelle2022/src/HOL/Examples/Interactive.thy",
                "port": 9000
            }
        }
    }
}
```

#### Start the PISA server
Finally, start a PISA server in a separate tmux window (similar to what was done above in Installation):
```bash
cd ~/Portal-to-ISAbelle
sbt "runMain pisa.server.PisaOneStageServer9000"
```
The port specified in the config (here `"port": 9000`) should match the number that appears in the command (`PisaOneStageServer9000`).

#### Run the eval!
Now try running the evaluation. An example script for running the evaluation is in `eval_scripts/eval_minif2f_isabelle_accelerate.sh`.
