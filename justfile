[group: 'test']
uncompress:
  tar -xvzf test-inputs.tar.gz

[group: 'test']
test: uncompress
  cargo nextest run --release

[group: 'test']
test_debug: uncompress
  cargo nextest run
