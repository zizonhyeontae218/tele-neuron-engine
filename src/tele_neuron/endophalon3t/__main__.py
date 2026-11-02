"""Run the historical training CLI with the owner-assigned 3T display name."""
from tele_neuron.endophalon3 import __main__ as historical
from . import variant_name

if __name__=='__main__':
    historical.variant_name=variant_name
    historical.main()
