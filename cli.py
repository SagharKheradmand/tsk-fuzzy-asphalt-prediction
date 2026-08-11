import numpy as np


def run_cli(models, scaler=None, input_dim=10):
    """
    Interactive command-line interface.
    """
    print("\nTSK Interface")
    print(f"Enter {input_dim} input values separated by space.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            s = input("x> ").strip()
        except EOFError:
            print("\nInput stream closed. Exiting CLI.")
            break
        if s.lower() == "exit":
            break

        try:
            values = np.array([float(v) for v in s.split()]).reshape(1, -1)

            if scaler:
                values = scaler.transform(values)

            for name, model in models.items():
                pred = model.predict(values)[0]
                print(f"{name}: {pred:.6f}")
            print()

        except Exception as e:
            print("Error:", e)
