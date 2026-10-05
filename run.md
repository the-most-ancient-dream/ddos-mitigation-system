# Run and test notes

Use the setup and launch instructions in [README.md](README.md).

## Dashboard simulation

Start the application with `--dashboard`, open the local dashboard, and press **Start simulated attack demo**. The demo posts generated statistics to the running detector; it does not send a network flood. It can add generated addresses to the blacklist.

## Automated tests

From the project root with the Python environment active:

```bash
python -m pytest tests/ -v
```

## Real packet simulation

`attack_simulator.py` sends packets. Only use it on an isolated lab interface and a target you own or are authorized to test. Do not run it against a public or production network.
