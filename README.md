# GLEXP427
Graph learning experiment notes and such.

# Prerequisites

## nvm install & such
#local dependencies installed on MacOS
```bash
SEASNet-14-49:~ pixel$ nvm install
No .nvmrc file found

Node Version Manager

Note: <version> refers to any version-like string nvm understands. This includes:
  - full or partial version numbers, starting with an optional "v" (0.10, v0.1.2, v1)
  - default (built-in) aliases: node, stable, unstable, iojs, system
  - custom aliases you define with `nvm alias foo`

 Any options that produce colorized output should respect the `--no-colors` option.

Usage:
  nvm --help                                Show this message
  nvm --version                             Print out the installed version of nvm
  nvm install [-s] <version>                Download and install a <version>, [-s] from source. Uses .nvmrc if available
    --reinstall-packages-from=<version>     When installing, reinstall packages installed in <node|iojs|node version number>
    --lts                                   When installing, only select from LTS (long-term support) versions
    --lts=<LTS name>                        When installing, only select from versions for a specific LTS line
    --skip-default-packages                 When installing, skip the default-packages file if it exists
    --latest-npm                            After installing, attempt to upgrade to the latest working npm on the given node version
  nvm uninstall <version>                   Uninstall a version
  nvm uninstall --lts                       Uninstall using automatic LTS (long-term support) alias `lts/*`, if available.
  nvm uninstall --lts=<LTS name>            Uninstall using automatic alias for provided LTS line, if available.
  nvm use [--silent] <version>              Modify PATH to use <version>. Uses .nvmrc if available
    --lts                                   Uses automatic LTS (long-term support) alias `lts/*`, if available.
    --lts=<LTS name>                        Uses automatic alias for provided LTS line, if available.
  nvm exec [--silent] <version> [<command>] Run <command> on <version>. Uses .nvmrc if available
    --lts                                   Uses automatic LTS (long-term support) alias `lts/*`, if available.
    --lts=<LTS name>                        Uses automatic alias for provided LTS line, if available.
  nvm run [--silent] <version> [<args>]     Run `node` on <version> with <args> as arguments. Uses .nvmrc if available
    --lts                                   Uses automatic LTS (long-term support) alias `lts/*`, if available.
    --lts=<LTS name>                        Uses automatic alias for provided LTS line, if available.
  nvm current                               Display currently activated version
  nvm ls                                    List installed versions
  nvm ls <version>                          List versions matching a given <version>
  nvm ls-remote                             List remote versions available for install
    --lts                                   When listing, only show LTS (long-term support) versions
  nvm ls-remote <version>                   List remote versions available for install, matching a given <version>
    --lts                                   When listing, only show LTS (long-term support) versions
    --lts=<LTS name>                        When listing, only show versions for a specific LTS line
  nvm version <version>                     Resolve the given description to a single local version
  nvm version-remote <version>              Resolve the given description to a single remote version
    --lts                                   When listing, only select from LTS (long-term support) versions
    --lts=<LTS name>                        When listing, only select from versions for a specific LTS line
  nvm deactivate                            Undo effects of `nvm` on current shell
  nvm alias [<pattern>]                     Show all aliases beginning with <pattern>
  nvm alias <name> <version>                Set an alias named <name> pointing to <version>
  nvm unalias <name>                        Deletes the alias named <name>
  nvm install-latest-npm                    Attempt to upgrade to the latest working `npm` on the current node version
  nvm reinstall-packages <version>          Reinstall global `npm` packages contained in <version> to current version
  nvm unload                                Unload `nvm` from shell
  nvm which [current | <version>]           Display path to installed node version. Uses .nvmrc if available
  nvm cache dir                             Display path to the cache directory for nvm
  nvm cache clear                           Empty cache directory for nvm

Example:
  nvm install 8.0.0                     Install a specific version number
  nvm use 8.0                           Use the latest available 8.0.x release
  nvm run 6.10.3 app.js                 Run app.js using node 6.10.3
  nvm exec 4.8.3 node app.js            Run `node app.js` with the PATH pointing to node 4.8.3
  nvm alias default 8.1.0               Set default node version on a shell
  nvm alias default node                Always default to the latest available node version on a shell

Note:
  to remove, delete, or uninstall nvm - just remove the `$NVM_DIR` folder (usually `~/.nvm`)
```

```bash
# install nvm
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.11/install.sh | bash


=> Close and reopen your terminal to start using nvm or run the following to use it now:

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm

# install node
nvm install node

=> Now using node v15.3.0 (npm v7.0.14)
Creating default alias: default -> node (-> v15.3.0)

# install local dependencies
nvm install

# run remaining setup
npm run build

# start web pack task in the background
npm run dev
```

```bash
# install webpack (required for npm run build)
npm install --save-dev web pack-cli
```

## python 3 & PsyTurk
### experiment file dir (MSLTM dir)
/Users/pixel/Documents/GLexp/mturk-statistical-learning-task-master
### venv dir
/Users/pixel/Documents/GLexp/psiturk
### make venv (using Python3 because 2 doesn't have venv)
python3 -m venv /Users/pixel/Documents/GLexp/psiturk
### activate it
source /Users/pixel/Documents/GLexp/psiturk/bin/activate
### install packages
pip install pandas networkx ipython bctpy

### Homebrew installing permissions:
(stackoverflow.com/questions/9800527)
#check permissions:
ls -ld /usr/local/Cellar
### change permission
sudo chgrp -R admin /usr/local/Cellar
### open it up
sudo chmod a+w /usr/local/Cellar

change ownership and permissions:
(bash)
```bash
You should probably change the ownership and permissions of /usr/local
back to your user account.
sudo chown -R $(whoami):admin /usr/local
```

# Piloting & Testing (editing files in MSLTM dir)
## Sandbox mode & Live mode
### Sandbox (psiturk or Mturk): locally or Haroku
home for database (where db is hosted): local (debugging), or Haroku (piloting, and formal run)
API: psiturk (for experimenter) or MTurk (participants will see)
HIT: MTurk; task to be put on MTurk

1) local sandbox HIT
2) Haroku sandbox HIT
3) Haroku live HIT

3 in common: task.js, custom.py

[config.txt] - database
1) change where we host database:

- edit experiment/config.txt: edit database_url (where we store the data).
  comment out line 17; uncomment line 18.

[setup_db.py] - 
python3 setup_db.py
(make sure there is the correct subjects.csv.gz, because .py is writing from it to participants.db.)

[psiturk (v2.3.11)] (may sure we are in experiment sub-folder)

(may need to install following packages on first run: pip3 install psycopg2-binary wheel python-Levenshtein)

(need to include AWS Access credentials in config.txt)

(in bash)
```bash
npm install (once?)

source /Users/pixel/Documents/GLexp/psiturk2/bin/activate
cd /Users/pixel/Documents/GLexp/mturk-graph-learning-427-master427/experiment
# below two lines enable npm command
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
npm run gulp #this updates task.js
psiturk
hit create
```
("num of participants?") 1

("reward") 1.00

("duration") 0.5

(URL: experiment!)

kill psiturk progress (if server is in blocked status):
```bash
pkill -f psiturk
```


To work locally: PsiTurk debugging in sandbox mode:
(bash)
debug:
If you type 'ipython -i custom.py" it should open up a new environment, kind of like typing v or nano
and Shift+Enter executes the code.

[variable/table names]
run reset_db.py and setup_db.py once right before mTurk, if not in debugging of course.
custom_models.py & config.txt: change the tablename to something like name_427
then:
reset_db.py: change the name of tables to the ones in custom_models.py & config.txt.
setup_db.py: now we can have raise Exception('Database is not empty!') since if reset_db, db will be empty.
  there may be a thing called schema that limits the change we have on tables.

[heroku debugging]
Error R10 (Boot timeout): tried to launch app, but failed in 3 min or something, it will kill the process.

database on heroku:
[bash]
heroku pg:psql
\dt # show database table
SELECT * FROM [table name] # look at specific table

debug on heroku but environment (step-by-step: (heroku exact is for online)
heroku run bash # heroku.py script is not run, just set up environment
ls
cd ./experiment
1. python -i custom.py # in python environment
2. psiturk
   server on
   debug
