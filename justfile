[group: 'test']
uncompress:
  tar -xvzf test-inputs.tar.gz
  python scripts/firrtl-module-splitter.py test-inputs/chipyard.harness.TestHarness.LargeBoomV3Config.fir  test-inputs/boom-modules
  python scripts/firrtl-module-splitter.py test-inputs/chipyard.harness.TestHarness.RocketConfig.fir       test-inputs/rocket-modules

[group: 'test']
test: uncompress
  cargo nextest run --release

[group: 'test']
test_debug: uncompress
  cargo nextest run
