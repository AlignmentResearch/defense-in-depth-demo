from hypothesis import settings

# For multi-GPU tests:
# - Increase the deadline because communication cost can be high
# - Derandomize inputs or else different processes may run different inputs at
#   once.
settings.register_profile("multigpu", deadline=5000, derandomize=True)
