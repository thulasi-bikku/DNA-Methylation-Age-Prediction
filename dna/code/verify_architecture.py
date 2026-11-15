"""Print the combiner type and parameter count of a trained Ludwig model directory."""
import sys
from ludwig.api import LudwigModel

m = LudwigModel.load(sys.argv[1])
print("combiner:", m.config["combiner"]["type"] if "combiner" in m.config else m.config.get("combiner"))
print("trainable parameters:", sum(p.numel() for p in m.model.parameters() if p.requires_grad))
