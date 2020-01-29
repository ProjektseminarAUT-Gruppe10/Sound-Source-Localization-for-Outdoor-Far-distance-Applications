import run_simulation

configs = run_simulation.load_configs('configs.json')

for config in configs:
    for i in range(0, config["runs"]):
        res = run_simulation.simulate(config)

        print(res)
