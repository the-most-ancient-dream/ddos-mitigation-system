# DDoS Detection and Mitigation

A local prototype that combines XDP/eBPF traffic monitoring, anomaly detection, an optional Random Forest classifier, and a Flask dashboard.

The included model is the balanced V2 binary classifier. The training CSV and generated evaluation graphs are not included.

## Run in Ubuntu or WSL2

Clone the repository into this folder name:

~~~bash
git clone https://github.com/ucs200525/rapid-corona.git ddos-mitigation-system
cd ddos-mitigation-system
~~~

Install the Linux eBPF tools and Python environment:

~~~bash
sudo apt update
sudo apt install -y python3-full python3-bpfcc build-essential
mkdir -p "$HOME/.venvs"
python3 -m venv --system-site-packages "$HOME/.venvs/ddos-mitigation-system"
source "$HOME/.venvs/ddos-mitigation-system/bin/activate"
python -m pip install -r requirements.txt
~~~

Start monitoring on the WSL network interface:

~~~bash
sudo "$HOME/.venvs/ddos-mitigation-system/bin/python" main.py --interface eth0 --mode generic --dashboard
~~~

The app auto-loads data/models/ddos_classifier_balanced_v2.joblib. Open http://localhost:5000 in Windows. Stop it with Ctrl+C in the Ubuntu terminal.

The dashboard uses Flask's development server without sign-in. Keep it on a trusted local test machine. WSL sees its own virtual network interface, not every Windows network adapter.

## Dashboard demo

The Start simulated attack demo button sends generated statistics to the app. It does not send a real network flood. It may add generated source IPs to the blacklist.

The separate attack_simulator.py sends network traffic. Use it only on a private lab interface and a target you own or are authorized to test.

## Retrain with the balanced dataset

The balanced CSV is not included. From WSL, a Windows D: drive is usually available under /mnt/d/. Save a new candidate separately so the included model stays available:

~~~bash
python -m src.ml.balanced_dataset_trainer \
  --dataset /mnt/d/capstone/datasets/balanced_dataset_v2.csv \
  --model-path data/models/ddos_classifier_balanced_v2_candidate.joblib \
  --results-dir reports/model_results_balanced_v2_candidate
~~~

The trainer fits preprocessing on its training split and writes model evaluation graphs and metrics to the results directory. To run a newly trained candidate, pass its path with --ml-model.

## Checks

Run the unit tests from the activated virtual environment:

~~~bash
python -m pytest tests/ -v
~~~

This prototype and its reported CSV metrics do not establish detection performance on live traffic. Review the limitations in the model evaluation output before relying on it.

There is no LICENSE file yet. Choose a license before publishing if you want others to reuse the code.