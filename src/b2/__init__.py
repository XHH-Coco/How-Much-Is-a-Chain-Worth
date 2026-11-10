"""How Much Is a Chain Worth? -- chain-information recovery in polymers.

Everything in this package that touches a number is deliberately stdlib-only, so
that the protocol layer (splits, Lambda, kappa, bootstrap) can be tested on a
machine where none of the three conda environments has been installed yet.
Third-party imports live only behind the arm implementations.
"""

__version__ = "0.1.0"

__all__ = ["config", "io", "protocol", "metrics", "analysis"]
