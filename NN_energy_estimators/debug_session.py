import torch
from loguru import logger
from ase.build import bulk
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running MatterSim on {device}")

si = bulk("Si", "diamond", a=5.43)
si.calc = MatterSimCalculator(device=device)
e = si.get_potential_energy()