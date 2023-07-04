# SPECIAL SETUP: Isabelle Proof Checker (WIP)


**A special setup is _required_ for tasks that use Isabelle proof checking:**

- `miniF2F_isabelle`

Follow this guide to set up Isabelle proof checking.

## Setup

The evaluation harness supports proof checking via [PISA](https://github.com/albertqjiang/Portal-to-ISAbelle/tree/56def2c39f85d211e1f40cc5765581a567879106).

Here are setup steps for a non-dockerized environment. The setup is heavily based on the [PISA readme](https://github.com/albertqjiang/Portal-to-ISAbelle/tree/56def2c39f85d211e1f40cc5765581a567879106)  and [Dockerfile](https://github.com/albertqjiang/Portal-to-ISAbelle/blob/main/docker/Dockerfile). You may need to refer to those if something goes wrong.

### Installation
```bash
sudo apt-get install zip
curl -s "https://get.sdkman.io" | bash
source "/home/seanw/.sdkman/bin/sdkman-init.sh"
sdk install java 11.0.11-open
sdk install sbt
sbt compile
sbt assembly

wget https://isabelle.in.tum.de/dist/Isabelle2022_linux.tar.gz && \
    tar -xzf Isabelle2022_linux.tar.gz

export WORK_DIR=/home/seanw

mv Isabelle2022 ${WORK_DIR}/

echo 'alias isabelle=${WORK_DIR}/Isabelle2022/bin/isabelle' >> ~/.bashrc
source ~/.bashrc

isabelle build -b -D ${WORK_DIR}/Isabelle2022/src/HOL/ -j 20
```

At the end, here's what the setup looks like:
- Portal-to-ISAbelle github repo in `~/Portal-to-ISAbelle`
- Isabelle in `~/Isabelle2022`
    - e.g. `ls ~/Isabelle2022` `->` \
      `ANNOUNCE  bin  contrib ...`
- Isabelle heaps in `~/.isabelle`
    - e.g. `ls ~/.isabelle/Isabelle2022/heaps/polyml-5.9_x86_64_32-linux/` `->` \
      `Group-Ring-Module          HOL-Corec_Examples     HOL-Isar_Examples                  HOL-Probability-ex ...`

To test it out: start the server
#### Start to the server
```bash
cd ~/Portal-to-ISAbelle
sbt "runMain pisa.server.PisaOneStageServer9000"
```

Now, move to the configuration steps below.

### Configuration

#### Set $PISA_PATH

Set the $PISA_PATH environment variable. TODO

#### Setup `Interactive.thy` file

TODO

#### Setup config
Specify a `isabelle_checker` field in the task's `config.json`.
Example (`configs/config/minif2f_isabelle.json`):
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

Start a PISA server in a separate tmux window using the following command (this was also done above in Installation):
```bash
cd ~/Portal-to-ISAbelle
sbt "runMain pisa.server.PisaOneStageServer9000"
```
- The port specified in the config (here `"port": 9000`) should match the number that appears in the command (`PisaOneStageServer9000`).

#### Run the eval!
Now try running the evaluation. An example script for running the evaluation is in `eval_scripts/eval_minif2f_isabelle_accelerate.sh`.